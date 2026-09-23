"""Original paper typography and marker encodings, with portable exports."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MplPath
import numpy as np
METHODS = ['Native','NAS','ENCORE','SPHERE','SADR']
MARKERS = dict(Native='o',NAS='D',ENCORE='s',SPHERE='^',SADR='v')
OUTPUT_DIR = Path('build/figures')
AUDITS = {}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def export(fig, stem, audit):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for ext in ['png','pdf','svg']:
        fig.savefig(OUTPUT_DIR / (stem+'.'+ext), dpi=600)
    AUDITS[stem] = dict(audit, width_inches=float(fig.get_size_inches()[0]), height_inches=float(fig.get_size_inches()[1]), font_family='STIXGeneral', dpi=600)
    plt.close(fig)


def configure():
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
        'font.weight':'normal','axes.labelweight':'normal','axes.titleweight':'normal',
        'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'legend.fontsize':9,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,'ytick.major.size':3,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
        'savefig.facecolor':'white','axes.unicode_minus':True})


def grid(ax):
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e2e2e2',lw=.45)


def endpoint_legend(fig):
    methods=[Line2D([],[],ls='none',marker=MARKERS[m],mfc='#bd532e' if m=='Native' else '#565656',
        mec='#bd532e' if m=='Native' else '#565656',ms=5.5,label=m) for m in METHODS]
    fig.legend(handles=methods,loc='upper center',bbox_to_anchor=(.54,.99),ncol=5,frameon=False,
               columnspacing=1.9,handletextpad=.5)
    editors=[Line2D([],[],ls='none',marker='o',mfc=c,mec='#565656',ms=5.5,label=l)
             for c,l in [('#565656','AlphaEdit'),('white','MEMIT')]]
    fig.legend(handles=editors,loc='upper center',bbox_to_anchor=(.54,.922),ncol=2,frameon=False,
               columnspacing=2,handletextpad=.5)


def require(ok, message):
    if not ok:
        raise ValueError(message)


@lru_cache(None)
def chord_marker(symbol):
    """Keep the native outline and append only its y=0 interior chord."""
    native = MarkerStyle(symbol)
    outline = native.get_path().transformed(native.get_transform())
    intersections = []
    for polygon in outline.to_polygons(closed_only=True):
        for a, b in zip(polygon[:-1], polygon[1:]):
            if a[1] == b[1] == 0:
                intersections.extend((a[0], b[0]))
            elif a[1] != b[1] and min(a[1], b[1]) <= 0 <= max(a[1], b[1]):
                intersections.append(a[0] - a[1] * (b[0] - a[0]) / (b[1] - a[1]))
    require(len(intersections) >= 2, f"No chord for {symbol}")
    chord = np.array([[min(intersections), 0.], [max(intersections), 0.]])
    interior = np.column_stack((np.linspace(chord[0, 0], chord[1, 0], 103)[1:-1], np.zeros(101)))
    require(outline.contains_points(interior, radius=1e-10).all(), f"Unclipped chord: {symbol}")
    compound = MplPath.make_compound_path(outline, MplPath(chord, [MplPath.MOVETO, MplPath.LINETO]))
    scale = 1 / MarkerStyle(compound).get_transform().get_matrix()[0, 0]
    return compound, scale, chord


def fill_for(model, dataset, order):
    if dataset == "CounterFact":
        return "lined" if model == "GPT-2 XL" else "none"
    if model == "GPT-2 XL":
        return "right"
    return "full" if order == "canonical" else "left"


def make_marker(symbol, color, fill, size, label=None):
    kw = dict(ls="", mec=color, mew=.8, label=label, alpha=.91)
    if fill == "lined":
        shape, correction, _ = chord_marker(symbol)
        kw.update(marker=shape, ms=size * correction, mfc="none")
    else:
        kw.update(marker=symbol, ms=size, fillstyle=fill,
                  mfc="none" if fill == "none" else color, markerfacecoloralt="white")
    return kw

