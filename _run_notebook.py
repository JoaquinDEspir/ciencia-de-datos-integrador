import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
path = Path("TP3_pipeline_mejorado.ipynb")
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=False)
try:
    client.execute()
except CellExecutionError as exc:
    path_error = path.with_name(path.stem + "_error.ipynb")
    nbformat.write(nb, path_error)
    raise SystemExit(f"Notebook executed with errors; see {path_error}") from exc
else:
    nbformat.write(nb, path)
    print("Notebook executed successfully and saved.")
