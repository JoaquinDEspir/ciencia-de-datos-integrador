# json_etl_dag.py
from __future__ import annotations

import os
import io
import csv
import json
import time
import hashlib
import datetime as dt
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import pandas as pd
import requests
import pendulum
from jsonschema import Draft202012Validator
from airflow.decorators import dag, task

# ──────────────────────────────────────────────────────────────────────────────
# RUTAS (dentro del contenedor de Astro/Airflow)
# ──────────────────────────────────────────────────────────────────────────────
BASE_PATH = "/usr/local/airflow/include"
RAW_PATH = f"{BASE_PATH}/data/raw/propiedades.json"           # Fallback local (único)
SCHEMA_PATH = f"{BASE_PATH}/schemas/propiedades.schema.json"  # JSON Schema ítem
OUT_DIR = f"{BASE_PATH}/data/processed"
CSV_OUT = f"{OUT_DIR}/propiedades_clean.csv"                  # Maestro consolidado (UPSERT)
PARQUET_OUT = f"{OUT_DIR}/propiedades_clean.parquet"
MANIFEST_OUT = f"{OUT_DIR}/manifest.json"
CHANGELOG_OUT = f"{OUT_DIR}/changelog.csv"                    # Bitácora de cambios

# ──────────────────────────────────────────────────────────────────────────────
# Fuente base por tipo (page la añade el fetch)
# ──────────────────────────────────────────────────────────────────────────────
BASE_URL_FMT = (
    "https://inmoup.com.ar/api/inmuebles/mendoza/ventas/{tipo}"
    "?precio[min]=5&precio[max]=12000000000000&limit=1000"
)
TIPOS = ["casas", "terrenos", "departamentos"]

# Env vars (opcionales)
MAX_ITEMS_ENV = "MAX_ITEMS"             # 0 o vacío => sin límite (default 0, aplica al total consolidado)
PAGE_SLEEP_ENV = "PAGE_SLEEP_SECS"      # pausa entre páginas (default 0.2s)
REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT" # timeout en segundos (default 60)
CSV_COLUMNS_ENV = "CSV_COLUMNS"         # columnas CSV (comma-separated)

DEFAULT_MAX_ITEMS = 0       # 0 = sin límite (sobre el total acumulado)
DEFAULT_PAGE_SLEEP = 0.2
DEFAULT_TIMEOUT = 60

# Columnas relevantes por defecto
DEFAULT_CSV_COLUMNS = [
    "propiedad_id", "tip_desc", "prp_dom", "loc_desc", "pro_desc",
    "con_desc", "grupo_tip_desc",
    "prp_pre_dol", "prp_pre",
    "banos", "dormitorios",
    "cocheras", "cochera",
    "sup_total", "sup_cubierta",
    "prp_alta", "prp_mod",
    "url_ficha_inmoup",
    "prp_lat", "prp_lng",
    "fotos_cantidad",
    # Si quisieras, podés agregar "categoria_inmoup" y setearla en transform.
]

# Clave primaria para upsert
PK_COL = "propiedad_id"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_items(obj: Any) -> List[Dict[str, Any]]:
    """Normaliza varios formatos de respuesta a lista de items."""
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], dict) and "pager" in obj["data"]:
            return obj["data"]["pager"] or []
        if "pager" in obj and isinstance(obj["pager"], list):
            return obj["pager"]
        if "items" in obj and isinstance(obj["items"], list):
            return obj["items"]
        return [obj]  # un único objeto
    if isinstance(obj, list):
        return obj
    return []

def _url_with_page(url: str, page: int) -> str:
    """Inserta/reemplaza el parámetro 'page' en la URL."""
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["page"] = str(page)
    new_query = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))

# ──────────────────────────────────────────────────────────────────────────────
# DAG con paralelismo: 3 fetch en paralelo → merge → validate → transform → upsert
# ──────────────────────────────────────────────────────────────────────────────
@dag(
    dag_id="json_etl_dag",
    start_date=pendulum.datetime(2025, 9, 1, 9, 0, tz="America/Argentina/Mendoza"),
    schedule="0 9 * * 1",   # lunes 09:00
    catchup=False,
    tags=["etl", "json", "inmoup", "weekly", "upsert", "pagination", "consolidated", "parallel"],
)
def json_ingestion_pipeline():
    """
    Flujo con paralelismo real:
      - fetch_json_tipo('casas'), fetch_json_tipo('terrenos'), fetch_json_tipo('departamentos') corren en paralelo
      - merge_raw consolida en un único payload
      - validate_json → transform → upsert_and_save generan un único CSV maestro
    """

    # Valores globales (para mantener consistencia en los 3 fetch)
    max_items_env = os.getenv(MAX_ITEMS_ENV, str(DEFAULT_MAX_ITEMS)).strip()
    sleep_secs_global = float(os.getenv(PAGE_SLEEP_ENV, str(DEFAULT_PAGE_SLEEP)))
    timeout_global = float(os.getenv(REQUEST_TIMEOUT_ENV, str(DEFAULT_TIMEOUT)))
    # 0/"" => ilimitado (sobre el total acumulado entre tipos)
    max_items_global = int(max_items_env) if max_items_env.isdigit() else DEFAULT_MAX_ITEMS

    @task
    def fetch_json_tipo(tipo: str) -> str:
        """
        Descarga paginada para un tipo dado (casas/terrenos/departamentos).
        Devuelve un JSON-STRING con:
            {
              "tipo": str,
              "meta": { "pages_fetched": int, "received_count": int, ... },
              "items_json": "[ ... ]"   # str JSON array de items
            }
        """
        os.makedirs(OUT_DIR, exist_ok=True)

        base_url = BASE_URL_FMT.format(tipo=tipo)
        sleep_secs = sleep_secs_global
        timeout = timeout_global

        items: List[Dict[str, Any]] = []
        seen_keys = set()
        pages_fetched = 0

        def add_batch(batch: List[Dict[str, Any]]) -> int:
            added = 0
            for rec in batch:
                pid = rec.get("propiedad_id")
                # dedupe robusto por id o hash
                key = ("pid", pid) if pid is not None else ("hash", hashlib.md5(
                    json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                items.append(rec)
                added += 1
            return added

        try:
            page = 1
            while True:
                url = _url_with_page(base_url, page)
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                obj = resp.json()

                batch = _extract_items(obj)
                if not batch:
                    break

                added = add_batch(batch)
                pages_fetched += 1
                print(f"[fetch_json_tipo] tipo={tipo} page={page} items_in_page={len(batch)} added={added} total_tipo={len(items)}")

                page += 1
                if sleep_secs > 0:
                    time.sleep(sleep_secs)

            meta = {
                "fetched_at_utc": dt.datetime.utcnow().isoformat() + "Z",
                "source": {"mode": "remote_url", "value": base_url},
                "pages_fetched": pages_fetched,
                "received_count": len(items),
                "tipo": tipo,
            }
            items_json = json.dumps(items, ensure_ascii=False)

        except Exception as e:
            # Fallback local SI y SOLO SI existe un único RAW_PATH (no por tipo)
            if os.path.exists(RAW_PATH):
                with open(RAW_PATH, "r", encoding="utf-8") as f:
                    local_text = f.read()
                try:
                    obj = json.loads(local_text)
                except Exception:
                    obj = local_text
                items = _extract_items(obj)
                meta = {
                    "fetched_at_utc": dt.datetime.utcnow().isoformat() + "Z",
                    "source": {"mode": "local_file", "value": RAW_PATH, "error_remote": str(e)},
                    "pages_fetched": 0,
                    "received_count": len(items),
                    "tipo": tipo,
                }
                items_json = json.dumps(items, ensure_ascii=False)
            else:
                raise

        # Devolvemos string JSON (menor overhead de XCom que objetos grandes)
        return json.dumps({"tipo": tipo, "meta": meta, "items_json": items_json}, ensure_ascii=False)

    @task
    def merge_raw(*partial_payloads: str) -> str:
        """
        Consolida los resultados de los 3 fetch en un único payload bruto.
        Respeta MAX_ITEMS_GLOBAL (si > 0) sobre el total.
        Devuelve JSON-STRING: {"meta": {...}, "raw": "{\"items\": [...]}"}  (raw es string)
        """
        all_items: List[Dict[str, Any]] = []
        seen_keys = set()
        pages_per_tipo: Dict[str, int] = {}
        received_per_tipo: Dict[str, int] = {}
        stopped_by_limit = False

        for pstr in partial_payloads:
            if not pstr:
                continue
            part = json.loads(pstr)
            tipo = part["tipo"]
            meta = part["meta"]
            items = json.loads(part["items_json"])

            # Merge meta per tipo
            pages_per_tipo[tipo] = meta.get("pages_fetched", 0)
            received_per_tipo[tipo] = meta.get("received_count", 0)

            # Merge de items con dedupe global
            for rec in items:
                pid = rec.get("propiedad_id")
                key = ("pid", pid) if pid is not None else ("hash", hashlib.md5(
                    json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_items.append(rec)

                # Aplicar límite global si corresponde
                if max_items_global > 0 and len(all_items) >= max_items_global:
                    stopped_by_limit = True
                    break
            if stopped_by_limit:
                break

        raw_text = json.dumps({"items": all_items}, ensure_ascii=False)
        raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        meta_out = {
            "fetched_at_utc": dt.datetime.utcnow().isoformat() + "Z",
            "source": {
                "mode": "remote_url",
                "value": "inmoup_parallel_consolidated",
                "tipos": TIPOS,
                "pages_per_tipo": pages_per_tipo,
                "received_per_tipo": received_per_tipo,
            },
            "raw_sha256": raw_hash,
            "received_count": len(all_items),
            "max_items": max_items_global,
            "stopped_by_limit": stopped_by_limit,
            "pages_fetched_total": sum(pages_per_tipo.values()),
        }
        print(f"[merge_raw] total_items={len(all_items)} pages_total={meta_out['pages_fetched_total']} stopped_by_limit={stopped_by_limit}")
        return json.dumps({"meta": meta_out, "raw": raw_text}, ensure_ascii=False)

    @task
    def validate_json(payload_json: str) -> str:
        bundle = json.loads(payload_json)
        raw_text = bundle["raw"]
        meta = bundle["meta"]

        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_item = json.load(f)

        data = json.loads(raw_text)
        items = _extract_items(data)

        validator = Draft202012Validator(schema_item)
        good: List[Dict[str, Any]] = []
        for rec in items:
            errs = sorted(validator.iter_errors(rec), key=lambda e: e.path)
            if not errs:
                good.append(rec)

        if not good:
            raise ValueError("Todos los registros fallaron la validación de schema.")

        meta_ext = {**meta, "validated_count": len(good)}
        return json.dumps({"meta": meta_ext, "valid_items": good}, ensure_ascii=False)

    @task
    def transform(valid_bundle_json: str) -> str:
        bundle = json.loads(valid_bundle_json)
        items: List[Dict[str, Any]] = bundle["valid_items"]

        def to_int(x):
            try:
                if x in (None, "", "null"):
                    return pd.NA
                return int(float(str(x).replace(",", ".")))
            except Exception:
                return pd.NA

        def to_float(x):
            try:
                if x in (None, "", "null"):
                    return pd.NA
                return float(str(x).replace(",", "."))  # acepta "123,45"
            except Exception:
                return pd.NA

        def to_str(x):
            if x is None:
                return None
            s = str(x).strip()
            return s if s != "" else None

        target_cols_env = os.getenv(CSV_COLUMNS_ENV)
        if target_cols_env:
            csv_cols = [c.strip() for c in target_cols_env.split(",") if c.strip()]
        else:
            csv_cols = DEFAULT_CSV_COLUMNS[:]

        out = {c: [] for c in csv_cols}

        def push(col, val):
            if col in out:
                out[col].append(val)

        for rec in items:
            propiedad_id = to_int(rec.get("propiedad_id"))
            tip_desc = to_str(rec.get("tip_desc"))
            prp_dom = to_str(rec.get("prp_dom"))
            loc_desc = to_str(rec.get("loc_desc"))
            pro_desc = to_str(rec.get("pro_desc"))
            con_desc = to_str(rec.get("con_desc"))
            grupo_tip_desc = to_str(rec.get("grupo_tip_desc"))

            prp_pre_dol = to_float(rec.get("prp_pre_dol"))
            prp_pre = to_float(rec.get("prp_pre"))

            banos = rec.get("banos", rec.get("baños"))
            banos = to_int(banos)
            dormitorios = to_int(rec.get("dormitorios"))

            cocheras = to_int(rec.get("cocheras"))
            cochera = to_int(rec.get("cochera"))

            sup_total = to_float(rec.get("sup_total"))
            sup_cubierta = to_float(rec.get("sup_cubierta"))

            prp_alta = to_str(rec.get("prp_alta"))
            prp_mod = to_str(rec.get("prp_mod"))
            url_ficha_inmoup = to_str(rec.get("url_ficha_inmoup"))

            prp_lat = to_str(rec.get("prp_lat"))
            prp_lng = to_str(rec.get("prp_lng"))

            fotos = rec.get("fotos") or []
            fotos_cantidad = int(len(fotos))

            # Si quisieras conservar el tipo de categoría:
            # categoria_inmoup = rec.get("grupo_tip_desc") or rec.get("tip_desc")  # ejemplo
            # push("categoria_inmoup", to_str(categoria_inmoup))

            push("propiedad_id", propiedad_id)
            push("tip_desc", tip_desc)
            push("prp_dom", prp_dom)
            push("loc_desc", loc_desc)
            push("pro_desc", pro_desc)
            push("con_desc", con_desc)
            push("grupo_tip_desc", grupo_tip_desc)
            push("prp_pre_dol", prp_pre_dol)
            push("prp_pre", prp_pre)
            push("banos", banos)
            push("dormitorios", dormitorios)
            push("cocheras", cocheras)
            push("cochera", cochera)
            push("sup_total", sup_total)
            push("sup_cubierta", sup_cubierta)
            push("prp_alta", prp_alta)
            push("prp_mod", prp_mod)
            push("url_ficha_inmoup", url_ficha_inmoup)
            push("prp_lat", prp_lat)
            push("prp_lng", prp_lng)
            push("fotos_cantidad", fotos_cantidad)

        df = pd.DataFrame(out)
        if PK_COL not in df.columns:
            raise ValueError(f"El CSV requiere la clave primaria '{PK_COL}' para upsert.")

        df = df[[c for c in csv_cols if c in df.columns]]
        df = df.dropna(subset=[PK_COL])
        try:
            df[PK_COL] = pd.to_numeric(df[PK_COL], errors="coerce").astype("Int64")
        except Exception:
            pass
        df = df.drop_duplicates(subset=[PK_COL], keep="first")

        buf = io.StringIO()
        df.to_json(buf, orient="records", force_ascii=False)

        stats = {
            "rows_after_transform": int(len(df)),
            "columns": df.columns.tolist(),
        }
        return json.dumps({"meta": bundle["meta"], "df_records_json": buf.getvalue(), "stats": stats}, ensure_ascii=False)

    @task(do_xcom_push=False)
    def upsert_and_save(transformed_json: str) -> None:
        bundle = json.loads(transformed_json)
        records = json.loads(bundle["df_records_json"])
        meta = bundle["meta"]
        stats = bundle["stats"]

        os.makedirs(OUT_DIR, exist_ok=True)
        df_new = pd.DataFrame(records)

        # === LECTURA CSV EXISTENTE CON SEP=';' ===
        if os.path.exists(CSV_OUT):
            df_old = pd.read_csv(CSV_OUT, dtype="unicode", sep=";")
            if PK_COL in df_old.columns:
                df_old[PK_COL] = pd.to_numeric(df_old[PK_COL], errors="coerce").astype("Int64")
        else:
            df_old = pd.DataFrame(columns=df_new.columns)

        # === Alineación de columnas ===
        all_cols = list(dict.fromkeys([*df_old.columns.tolist(), *df_new.columns.tolist()]))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)

        df_old_idx = df_old.set_index(PK_COL, drop=False)
        df_new_idx = df_new.set_index(PK_COL, drop=False)

        existing_ids = set(df_old_idx.index.dropna().tolist())
        incoming_ids = set(df_new_idx.index.dropna().tolist())

        to_insert = sorted(list(incoming_ids - existing_ids))
        intersect = sorted(list(incoming_ids & existing_ids))

        cols_to_compare = [c for c in all_cols if c != PK_COL]
        changed_ids: List[int] = []
        for _id in intersect:
            old_row = df_old_idx.loc[_id:_id, cols_to_compare]
            new_row = df_new_idx.loc[_id:_id, cols_to_compare]
            if not old_row.equals(new_row):
                changed_ids.append(_id)

        df_out = df_old_idx.copy()
        if to_insert:
            df_out = pd.concat([df_out, df_new_idx.loc[to_insert]], axis=0)
        if changed_ids:
            df_out.loc[changed_ids] = df_new_idx.loc[changed_ids]

        df_out = df_out.sort_index().reset_index(drop=True)

        # === ESCRITURA CSV CON SEP=';' Y BOM PARA EXCEL ===
        df_out.to_csv(
            CSV_OUT,
            index=False,
            sep=";",
            encoding="utf-8-sig",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )

        # Parquet opcional
        try:
            df_out.to_parquet(PARQUET_OUT, index=False)
        except Exception as e:
            print(f"[upsert_and_save] Parquet no disponible: {e}")

        now_utc = dt.datetime.utcnow().isoformat() + "Z"
        changes = []
        for _id in to_insert:
            changes.append({"ts_utc": now_utc, "action": "insert", PK_COL: _id})
        for _id in changed_ids:
            changes.append({"ts_utc": now_utc, "action": "update", PK_COL: _id})

        # === CHANGELOG TAMBIÉN CON SEP=';' ===
        if changes:
            df_chg = pd.DataFrame(changes)
            if os.path.exists(CHANGELOG_OUT):
                df_prev = pd.read_csv(CHANGELOG_OUT, dtype="unicode", sep=";")
                df_chg = pd.concat([df_prev, df_chg], ignore_index=True)
            df_chg.to_csv(
                CHANGELOG_OUT,
                index=False,
                sep=";",
                encoding="utf-8-sig",
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )

        with open(CSV_OUT, "rb") as f:
            csv_hash = hashlib.sha256(f.read()).hexdigest()

        manifest = {
            "generated_at_utc": now_utc,
            "source": meta["source"],
            "raw_sha256": meta["raw_sha256"],
            "csv_path": CSV_OUT,
            "parquet_path": PARQUET_OUT if os.path.exists(PARQUET_OUT) else None,
            "csv_sha256": csv_hash,
            "row_count": int(len(df_out)),
            "columns": df_out.columns.tolist(),
            "stats": stats,
            "max_items": meta.get("max_items"),
            "received_count": meta.get("received_count"),
            "validated_count": meta.get("validated_count"),
            "stopped_by_limit": meta.get("stopped_by_limit"),
            "pages_fetched_total": meta.get("pages_fetched_total"),
            "pages_per_tipo": meta["source"].get("pages_per_tipo") if isinstance(meta.get("source"), dict) else None,
            "received_per_tipo": meta["source"].get("received_per_tipo") if isinstance(meta.get("source"), dict) else None,
            "upsert": {
                "inserted": len(to_insert),
                "updated": len(changed_ids),
                "existing_before": int(len(df_old)),
                "existing_after": int(len(df_out)),
            },
        }
        with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(
            f"[upsert_and_save] maestro={CSV_OUT} total={manifest['row_count']} "
            f"ins={manifest['upsert']['inserted']} upd={manifest['upsert']['updated']} "
            f"pages_total={manifest['pages_fetched_total']} sha256={csv_hash[:12]}..."
        )

    # ── Definición del grafo de tareas ─────────────────────────────────────────
    # 3 fetch en paralelo
    f_casas = fetch_json_tipo.override(task_id="fetch_json_casas")("casas")
    f_terrenos = fetch_json_tipo.override(task_id="fetch_json_terrenos")("terrenos")
    f_deptos = fetch_json_tipo.override(task_id="fetch_json_departamentos")("departamentos")

    # merge (espera a los 3)
    merged = merge_raw(f_casas, f_terrenos, f_deptos)

    # pipeline único
    validated = validate_json(merged)
    transformed = transform(validated)
    upsert_and_save(transformed)

json_ingestion_pipeline()
