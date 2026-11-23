"""Met à jour les cellules Markdown du notebook exécuté sans le réexécuter.

Usage : python tools/inject_markdown.py
Le code des cellules doit être identique (vérifié) : seules les cellules de texte sont remplacées.
"""
import json
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Sylla_Nfaly_BigData_KMeans.ipynb"
executed = json.loads(NB.read_text(encoding="utf-8"))
shutil.copy(NB, ROOT / "executed_backup.ipynb")
runpy.run_path(str(ROOT / "tools" / "build_notebook.py"))          # réécrit la version non exécutée
fresh = json.loads(NB.read_text(encoding="utf-8"))
if len(fresh["cells"]) != len(executed["cells"]):
    shutil.copy(ROOT / "executed_backup.ipynb", NB)
    sys.exit("nombre de cellules différent : réexécuter le notebook")
for i, (new, old) in enumerate(zip(fresh["cells"], executed["cells"])):
    if new["cell_type"] == "markdown":
        old["source"] = new["source"]
    elif new["source"] != old["source"]:
        shutil.copy(ROOT / "executed_backup.ipynb", NB)
        sys.exit(f"code différent dans la cellule {i} : réexécuter le notebook")
NB.write_text(json.dumps(executed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("cellules Markdown mises à jour dans", NB.name)
