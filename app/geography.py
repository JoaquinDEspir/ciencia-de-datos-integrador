"""Utility helpers to manage Mendoza neighbourhood labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEIGHBORHOODS_PATH = (
    PROJECT_ROOT / "include" / "data" / "reference" / "neighborhoods_by_loc.json"
)


Bounds = Dict[str, float]
Rule = Dict[str, object]


def _rule(bounds: Bounds, name: str) -> Rule:
    return {"bounds": bounds, "name": name}


HEURISTIC_RULES: Dict[str, List[Rule]] = {
    "capital": [
        _rule({"lat_max": -32.895}, "Microcentro"),
        _rule({"lat_min": -32.895, "lat_max": -32.885}, "La Quinta Seccion"),
        _rule({"lat_min": -32.885, "lat_max": -32.875}, "La Cuarta Seccion"),
        _rule({"lat_min": -32.875}, "La Sexta Seccion"),
        _rule({"lat_min": -32.905}, "La Favorita"),
    ],
    "godoy cruz": [
        _rule({"lat_max": -32.94}, "Villa Hipodromo"),
        _rule({"lat_min": -32.94, "lat_max": -32.915}, "Godoy Cruz Centro"),
        _rule({"lat_min": -32.915}, "Benegas"),
    ],
    "guaymallen": [
        _rule({"lat_max": -32.92}, "Rodeo de la Cruz"),
        _rule({"lng_min": -68.78}, "Dorrego"),
        _rule({"lng_max": -68.74}, "Bermejo"),
        _rule({}, "Villa Nueva"),
    ],
    "maipu": [
        _rule({"lat_max": -33.02}, "Gutierrez"),
        _rule({"lng_min": -68.80}, "Coquimbito"),
        _rule({"lng_max": -68.75}, "Luzuriaga"),
        _rule({}, "Ciudad de Maipu"),
    ],
    "lujan de cuyo": [
        _rule({"lat_max": -33.10}, "Agrelo"),
        _rule({"lat_min": -33.10, "lat_max": -33.02}, "Ciudad de Lujan"),
        _rule({"lat_min": -33.03, "lat_max": -32.97}, "Chacras de Coria"),
        _rule({"lat_min": -33.01, "lat_max": -32.97}, "Carrodilla"),
        _rule({"lat_min": -33.02}, "Vistalba"),
        _rule({"lat_min": -32.98, "lng_min": -68.82}, "La Puntilla"),
    ],
    "las heras": [
        _rule({"lat_max": -32.90}, "El Algarrobal"),
        _rule({"lat_min": -32.90, "lat_max": -32.84}, "Ciudad de Las Heras"),
        _rule({"lat_min": -32.84}, "El Challao"),
    ],
}


def _load_reference_mapping() -> Dict[str, List[str]]:
    if NEIGHBORHOODS_PATH.exists():
        return json.loads(NEIGHBORHOODS_PATH.read_text(encoding="utf-8-sig"))
    return {}


REFERENCE_MAPPING_RAW = _load_reference_mapping()
REFERENCE_MAPPING: Dict[str, List[str]] = {
    key.lower(): list(values) for key, values in REFERENCE_MAPPING_RAW.items()
}
REFERENCE_CANONICAL: Dict[str, str] = {
    key.lower(): key for key in REFERENCE_MAPPING_RAW.keys()
}


def _matches(bounds: Bounds, lat: float, lng: float) -> bool:
    lat_min = bounds.get("lat_min", -math.inf)
    lat_max = bounds.get("lat_max", math.inf)
    lng_min = bounds.get("lng_min", -math.inf)
    lng_max = bounds.get("lng_max", math.inf)
    return lat_min <= lat <= lat_max and lng_min <= lng <= lng_max


def _canonical_location(loc: str) -> str:
    norm = loc.strip().lower()
    return REFERENCE_CANONICAL.get(norm, loc.strip() or "Sin localidad")


def _fallback_label(loc: str, suffix: str) -> str:
    canonical = _canonical_location(loc)
    return f"{canonical} - {suffix}"


def infer_neighborhood(loc: str, lat: float | None, lng: float | None) -> str:
    """Return a neighbourhood label for the given location and coordinates."""
    canonical_loc = _canonical_location(loc)
    norm_loc = canonical_loc.lower()

    if lat is None or lng is None or math.isnan(lat) or math.isnan(lng):
        return _fallback_label(canonical_loc, "Sin Coordenadas")

    for rule in HEURISTIC_RULES.get(norm_loc, []):
        if _matches(rule.get("bounds", {}), lat, lng):
            candidate = f"{canonical_loc} - {rule['name']}"
            reference = REFERENCE_MAPPING.get(norm_loc)
            if not reference or candidate in reference:
                return candidate
            return candidate

    reference = REFERENCE_MAPPING.get(norm_loc)
    if reference:
        return reference[0]

    return _fallback_label(canonical_loc, "Otros")


def list_neighborhoods(loc: str) -> List[str]:
    """List known neighbourhood labels for a locality, including fallbacks."""
    canonical_loc = _canonical_location(loc)
    norm_loc = canonical_loc.lower()

    entries: List[str] = []
    reference = REFERENCE_MAPPING.get(norm_loc)
    if reference:
        entries.extend(reference)
    else:
        entries.extend(
            f"{canonical_loc} - {rule['name']}"
            for rule in HEURISTIC_RULES.get(norm_loc, [])
        )
    # Ensure fallbacks exist
    for suffix in ("Otros", "Sin Coordenadas"):
        label = _fallback_label(canonical_loc, suffix)
        if label not in entries:
            entries.append(label)

    # Deduplicate preserving order
    seen = set()
    result: List[str] = []
    for label in entries:
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def all_neighborhoods() -> List[str]:
    """Return all known neighbourhood labels."""
    labels: List[str] = []
    if REFERENCE_MAPPING:
        for loc_labels in REFERENCE_MAPPING.values():
            labels.extend(loc_labels)
    else:
        for loc in HEURISTIC_RULES.keys():
            labels.extend(list_neighborhoods(loc))
    seen = set()
    unique: List[str] = []
    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)
    return unique
