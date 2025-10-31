import json
from pathlib import Path

path = Path("Modelado_y_Evaluacion_propiedades.ipynb")
data = json.loads(path.read_text(encoding="utf-8"))

for idx, cell in enumerate(data["cells"]):
    if cell.get("outputs"):
        print(f"Cell {idx} outputs:")
        for out in cell["outputs"]:
            if out.get("output_type") == "execute_result":
                text = ''.join(out.get("data", {}).get("text/plain", ""))
                print(text)
            elif out.get("output_type") == "stream":
                print(''.join(out.get("text", "")))
            elif out.get("output_type") == "error":
                print("ERROR:", out.get("ename"), out.get("evalue"))
        print("-" * 80)
