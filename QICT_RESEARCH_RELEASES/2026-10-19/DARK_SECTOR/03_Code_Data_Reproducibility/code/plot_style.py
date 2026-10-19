"""Shared vector-figure typography, contrast and series order."""
from cycler import cycler
import matplotlib as mpl
from pathlib import Path
import io, os

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.6,
    'axes.prop_cycle': cycler(color=['#1F6F5F', '#555555', '#94948E']),
    'legend.frameon': False, 'legend.fontsize': 9,
    'grid.color': '#D8D8D4', 'grid.linewidth': .5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 160,
    'savefig.facecolor': 'white', 'axes.facecolor': 'white',
})


def save_figure(figure, path, **kwargs):
    """Publish a complete vector file atomically, after rendering it in memory."""
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer=io.BytesIO()
    figure.savefig(buffer,format=path.suffix.lstrip('.'),**kwargs)
    staged=path.with_name(path.name+'.staged')
    with staged.open('wb') as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged,path)
