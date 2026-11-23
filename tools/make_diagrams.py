"""Génère les schémas draw.io du projet puis les exporte en PNG (notebook) et PDF (LaTeX).

Usage : python tools/make_diagrams.py
"""
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagrams"
OUT.mkdir(exist_ok=True)
DRAWIO = r"C:\Program Files\draw.io\draw.io.exe"

FONT = "fontFamily=Helvetica;fontColor=#1E293B;"
ROLE = {
    "data": "fillColor=#DBEAFE;strokeColor=#2563EB;",
    "op": "fillColor=#F1F5F9;strokeColor=#475569;",
    "state": "fillColor=#EDE9FE;strokeColor=#7C3AED;",
    "out": "fillColor=#D1FAE5;strokeColor=#059669;",
    "note": "fillColor=#FFF7ED;strokeColor=#F27100;",
    "hot": "fillColor=#FEE2E2;strokeColor=#DC2626;",
    "head": "fillColor=#1E293B;strokeColor=#1E293B;fontColor=#FFFFFF;",
    "panel": "fillColor=none;strokeColor=#CBD5E1;dashed=1;",
}


class Diagram:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.cells, self.n = [], 2

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, label, x, y, w, h, role="op", size=13, extra=""):
        cid = self._id()
        style = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;{FONT}fontSize={size};strokeWidth=1.5;"
                 f"{ROLE[role]}{extra}")
        self.cells.append(f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
                          f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def text(self, label, x, y, w, h, size=13, align="center", color="#1E293B", bold=False):
        cid = self._id()
        style = (f"text;html=1;whiteSpace=wrap;{FONT}fontSize={size};align={align};verticalAlign=middle;"
                 f"fontColor={color};{'fontStyle=1;' if bold else ''}")
        self.cells.append(f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
                          f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def edge(self, src, tgt, label="", dashed=False, color="#475569", exit=None, entry=None, points=None):
        cid = self._id()
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeWidth=1.6;"
                 f"strokeColor={color};{FONT}fontSize=11;labelBackgroundColor=#FFFFFF;")
        if dashed:
            style += "dashed=1;dashPattern=6 4;"
        if exit:
            style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = ""
        if points:
            pts = "<Array as=\"points\">" + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>"
        self.cells.append(f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" edge="1" '
                          f'parent="1" source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
        return cid

    def save(self):
        xml = (f'<mxfile host="tools/make_diagrams.py"><diagram name="{self.name}" id="{self.name}">'
               f'<mxGraphModel dx="{self.width}" dy="{self.height}" grid="0" gridSize="10" guides="1" tooltips="1" '
               f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.width}" '
               f'pageHeight="{self.height}" background="#FFFFFF" math="0" shadow="0"><root><mxCell id="0"/>'
               f'<mxCell id="1" parent="0"/>{"".join(self.cells)}</root></mxGraphModel></diagram></mxfile>')
        path = OUT / f"{self.name}.drawio"
        path.write_text(xml, encoding="utf-8")
        return path


# ------------------------------------------------------------------ 00 vue d'ensemble
d = Diagram("00_overview", 1640, 470)
d.text("<b>Un même objectif, regrouper des points sans étiquette : cinq jeux de contraintes</b>", 20, 10, 1600, 36, size=18)
cols = [
    ("A · Séquentiel", "Python<br>un point à la fois", "O(k·d), constante", "aucun : le point i dépend du point i−1", "data"),
    ("B · Streaming", "Python + scikit-learn<br>T derniers batches, poids r<sup>t</sup>", "O(T·b·d), bornée", "aucun entre batches", "data"),
    ("C · Distribué", "Apache Beam<br>Map + CombinePerKey + side input", "données réparties<br>k·d diffusé à chaque worker", "très bon : agrégation combinable", "out"),
    ("D · Séquentiel avec état", "Apache Beam<br>DoFn avec état, 1 clé ou K′ clés", "O(k·d) par clé", "1 clé = 1 worker<br>K′ clés puis fusion", "state"),
    ("E · Streaming distribué", "Apache Beam<br>TestStream, fenêtres, état des T batches", "O(T·b·d) par clé", "entre flux indépendants", "state"),
]
for i, (title, how, mem, par, role) in enumerate(cols):
    x = 20 + i * 324
    d.box(f"<b>{title}</b>", x, 60, 300, 50, "head", size=16)
    d.box(how, x, 122, 300, 78, role, size=14)
    d.text("MÉMOIRE", x, 212, 300, 22, size=11, color="#C2410C", bold=True)
    d.box(mem, x, 236, 300, 64, "op", size=14)
    d.text("PARALLÉLISME", x, 312, 300, 22, size=11, color="#C2410C", bold=True)
    d.box(par, x, 336, 300, 64, "op", size=14)
d.text("k clusters · d dimensions · T batches gardés · b points par batch · K′ clés", 20, 418, 1600, 30, size=13, color="#64748B")
d.save()

# ------------------------------------------------------------------ 01 fenêtre glissante (B)
d = Diagram("01_streaming_window", 1500, 470)
d.text("<b>Partie B : on garde les T derniers batches, le batch d'âge t pèse r<sup>t</sup></b>", 20, 8, 1460, 34, size=18)
opac = [35, 50, 65, 82, 100]
ids = []
for i in range(5):
    t = 4 - i
    x = 250 + i * 190
    label = f"<b>B<sup>{t}</sup></b><br>âge t = {t}<br>poids r<sup>{t}</sup>" if t else "<b>B<sup>0</sup></b><br>nouveau batch<br>poids r<sup>0</sup> = 1"
    ids.append(d.box(label, x, 70, 160, 96, "data", size=14, extra=f"opacity={opac[i]};"))
d.text("fenêtre X (au plus T = 5 batches)", 250, 170, 920, 26, size=13, color="#64748B")
gone = d.box("plus vieux batch<br>retiré si |X| = T", 20, 82, 180, 72, "hot", size=13)
d.edge(ids[0], gone)
new = d.box("batch qui arrive", 1240, 92, 200, 52, "data", size=14)
d.edge(new, ids[-1])
init = d.box("<b>initialisation</b><br>barycentres pondérés<br>de la partition P", 250, 240, 300, 96, "note", size=14)
fit = d.box("<b>k-means pondéré</b><br>KMeans(init=C, n_init=1)<br>.fit(X, sample_weight=r<sup>t</sup>)", 620, 240, 330, 96, "op", size=14)
out = d.box("<b>sortie</b><br>centres C<br>partition P", 1020, 240, 220, 96, "out", size=14)
d.edge(init, fit)
d.edge(fit, out)
d.edge(out, init, "P réutilisée au batch suivant", dashed=True, color="#F27100", exit=(0.5, 1), entry=(0.5, 1),
       points=[(1130, 390), (400, 390)])
d.text("Avec w<sub>t</sub> = r<sup>t</sup> et r &lt; 1, un r petit oublie vite l'historique ; r &gt; 1 favoriserait les vieux batches.",
       250, 410, 990, 40, size=13, color="#64748B")
d.save()

# ------------------------------------------------------------------ 02 pipeline distribué (C)
d = Diagram("02_beam_distributed", 1640, 640)
d.text("<b>Partie C : Lloyd distribué, n itérations dépliées dans un seul DAG</b>", 20, 8, 1600, 34, size=18)
r1 = [d.box("ReadFromText<br>(un point par ligne)", 20, 70, 210, 70, "data"),
      d.box("Map<br>parse → (id, coordonnées)", 270, 70, 230, 70, "op"),
      d.box("Map<br>clé = hash(id, graine) mod k", 540, 70, 250, 70, "op"),
      d.box("CombinePerKey(MeanFn)<br>.with_hot_key_fanout(4)<br>accumulateur (somme, effectif)", 830, 64, 290, 82, "op"),
      d.box("<b>centroids₀</b><br>(cluster, centre)", 1160, 70, 200, 70, "out")]
for a, b in zip(r1, r1[1:]):
    d.edge(a, b)
d.text("C.1", 20, 44, 60, 22, size=12, color="#C2410C", bold=True)
d.text("C.2", 540, 44, 60, 22, size=12, color="#C2410C", bold=True)
d.text("C.3", 830, 44, 60, 22, size=12, color="#C2410C", bold=True)
pts = d.box("points<br>(même PCollection)", 270, 230, 230, 70, "data")
asg = d.box("<b>Map(assign_cluster)</b><br>→ (cluster le plus proche, point)", 540, 230, 290, 70, "op")
comb = d.box("CombinePerKey(MeanFn)<br>(somme, effectif) par cluster", 880, 230, 260, 70, "op")
ci = d.box("<b>centroids<sub>i</sub></b>", 1190, 230, 170, 70, "out")
d.text("C.4 · itération i = 1 … n", 540, 204, 300, 22, size=12, color="#C2410C", bold=True, align="left")
d.edge(r1[1], pts, exit=(0.5, 1), entry=(0.5, 0))
d.edge(pts, asg)
d.edge(asg, comb)
d.edge(comb, ci)
d.edge(r1[4], asg, "side input AsDict(centroids<sub>i−1</sub>)", dashed=True, color="#F27100", exit=(0.5, 1), entry=(0.5, 0),
       points=[(1260, 180), (685, 180)])
d.edge(ci, asg, "itération suivante", dashed=True, color="#F27100", exit=(0.5, 1), entry=(0.5, 1),
       points=[(1275, 340), (685, 340)])
d.box("boucle dépliée : A₁ B₁ A₂ B₂ … dans le même graphe<br>(un DAG ne peut pas contenir de cycle, cours diap. 12-13)",
      1390, 215, 230, 100, "note", size=12)
d.text("<b>Version optimisée (C.4.ii)</b>", 20, 392, 600, 28, size=15, align="left")
o = [d.box("BatchElements<br>paquets de 512 à 8192 points", 20, 430, 250, 76, "data"),
     d.box("<b>FlatMap</b><br>distances NumPy vectorisées<br>→ ≤ k sommes partielles par paquet", 310, 430, 330, 76, "op"),
     d.box("CombinePerKey(SumCountFn)<br>fusion des (somme, effectif)", 680, 430, 280, 76, "op"),
     d.box("<b>centroids<sub>i</sub></b>", 1000, 430, 170, 76, "out")]
for a, b in zip(o, o[1:]):
    d.edge(a, b)
mat = d.box("matrice k × d<br>AsSingleton", 310, 548, 200, 60, "note", size=13)
d.edge(mat, o[1], dashed=True, color="#F27100", exit=(0.5, 0), entry=(0.3, 1))
d.text("n éléments envoyés à l'agrégation → environ k × (nombre de paquets)", 560, 548, 610, 60, size=13, color="#64748B", align="left")
d.save()

# ------------------------------------------------------------------ 03 état (D)
d = Diagram("03_beam_stateful", 1640, 600)
d.text("<b>Partie D : l'algorithme de A dans un DoFn avec état</b>", 20, 8, 1600, 34, size=18)
d.box("", 20, 56, 720, 520, "panel")
d.text("<b>Clé unique</b>", 40, 66, 300, 28, size=15, align="left")
p1 = d.box("points", 50, 120, 150, 60, "data")
k1 = d.box("Map<br>(0, point)", 240, 120, 150, 60, "op")
s1 = d.box("<b>DoFn avec état</b><br>BagState {centres, effectifs, vus}<br>macqueen_update (partie A)", 430, 104, 290, 92, "state")
d.edge(p1, k1)
d.edge(k1, s1)
hot = d.box("un seul worker : tous les points traités en série<br>et envoyés à la même machine", 430, 230, 290, 70, "hot", size=12)
d.edge(s1, hot, dashed=True, color="#DC2626", exit=(0.5, 1), entry=(0.5, 0))
t1 = d.box("<b>copie de l'état à chaque point</b> (sortie secondaire, avec n_vus)<br>Top.PerKey(1, n_vus) → état final, quel que soit l'ordre", 240, 340, 480, 70, "note", size=13)
d.edge(hot, t1, exit=(0.5, 1), entry=(0.75, 0))
o1 = d.box("<b>centres finaux</b>", 240, 450, 480, 56, "out")
d.edge(t1, o1)
d.text("ordre d'arrivée non garanti : les k premiers points <i>reçus</i> deviennent les centres", 40, 520, 680, 44, size=12, color="#64748B")

d.box("", 780, 56, 840, 520, "panel")
d.text("<b>K′ clés, puis fusion</b>", 800, 66, 300, 28, size=15, align="left")
p2 = d.box("points", 810, 250, 130, 60, "data")
k2 = d.box("Map<br>(hash(id) mod K′, point)", 970, 250, 190, 60, "op")
d.edge(p2, k2)
init = d.box("init commune<br>Top.Smallest(k) → AsSingleton", 1190, 104, 260, 60, "note", size=13)
states = []
for i, lab in enumerate(("clé 0", "clé 1", "clé K′−1")):
    sid = d.box(f"<b>DoFn avec état</b> · {lab}", 1190, 196 + i * 82, 260, 58, "state", size=13)
    states.append(sid)
    d.edge(k2, sid, exit=(1, 0.5), entry=(0, 0.5), points=[(1175, 280), (1175, 225 + i * 82)])
d.edge(init, states[0], "même init pour toutes les clés", dashed=True, color="#F27100")
d.text("en parallèle", 1190, 440, 260, 24, size=12, color="#059669", bold=True)
m = d.box("<b>fusion</b><br>μ<sub>j</sub> = Σ<sub>s</sub> n<sub>j</sub><sup>s</sup> μ<sub>j</sub><sup>s</sup> / Σ<sub>s</sub> n<sub>j</sub><sup>s</sup>", 1190, 478, 260, 76, "out", size=13)
for i, sid in enumerate(states):
    d.edge(sid, m, exit=(1, 0.5), entry=(1, 0.5), points=[(1500 + 18 * i, 225 + i * 82), (1500 + 18 * i, 516)])
d.text("approximation : chaque fragment ne voit que n/K′ points", 800, 500, 370, 50, size=12, color="#64748B", align="left")
d.save()

# ------------------------------------------------------------------ 04 streaming (E)
d = Diagram("04_beam_streaming", 1640, 470)
d.text("<b>Partie E : la partie B dans Beam, avec TestStream, fenêtres et état</b>", 20, 8, 1600, 34, size=18)
r = [d.box("<b>TestStream</b><br>points horodatés<br>(batch t = minute t)", 20, 70, 220, 84, "data"),
     d.box("WindowInto<br>FixedWindows(60 s)<br>1 fenêtre = 1 batch", 280, 70, 220, 84, "op"),
     d.box("GroupByKey<br>batch complet quand le<br>watermark passe la fenêtre", 540, 70, 250, 84, "op"),
     d.box("batch_id = début<br>de la fenêtre / 60", 830, 70, 200, 84, "op"),
     d.box("WindowInto<br>GlobalWindows<br>(l'état vit entre batches)", 1070, 70, 240, 84, "op")]
for a, b in zip(r, r[1:]):
    d.edge(a, b)
late = d.box("points en retard (fenêtre expirée) : écartés par Beam", 540, 178, 250, 56, "hot", size=12)
d.edge(late, r[2], dashed=True, color="#DC2626", exit=(0.5, 0), entry=(0.5, 1))
st = d.box("<b>DoFn avec état</b><br>BagState : les T derniers batches<br>rangés par batch_id", 1070, 270, 300, 92, "state")
d.edge(r[4], st, exit=(0.5, 1), entry=(0.5, 0))
up = d.box("<b>streaming_update</b><br>la fonction de la partie B<br>(même code, même résultat)", 690, 270, 300, 92, "op")
d.edge(st, up)
outb = d.box("<b>batch_XXX.json</b><br>centres, partition<br>(sortie idempotente)", 330, 270, 280, 92, "out")
d.edge(up, outb)
d.box("batch déjà vu : ignoré<br>batch trop ancien : écarté<br>(idempotence)", 1400, 270, 220, 92, "note", size=12)
d.text("une clé = un flux = un worker ; on parallélise entre flux indépendants (un modèle par capteur)", 330, 392, 1290, 40,
       size=13, color="#64748B", align="left")
d.save()

# ------------------------------------------------------------------ export
for f in sorted(OUT.glob("*.drawio")):
    for fmt, extra in (("png", ["-s", "2"]), ("pdf", ["--crop"])):
        target = f.with_suffix(f".{fmt}")
        subprocess.run([DRAWIO, "-x", "-f", fmt, *extra, "-o", str(target), str(f)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("exporté :", target.name)
