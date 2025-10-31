import json
from pathlib import Path

data = json.loads(Path("Modelado_y_Evaluacion_propiedades.ipynb").read_text(encoding="utf-8"))
print(''.join(data['cells'][18]['source']))
