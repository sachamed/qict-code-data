"""Atomically publish complete Matplotlib files to the document directories."""
from pathlib import Path
import os
import shutil
import tempfile

def save_figure_atomic(figure, destination, **kwargs):
    """Render away from the destination, validate PDF completeness, then replace.

    Resolving a legacy source-view symlink preserves its canonical document
    target. A reader sees either the previous complete file or the new one.
    """
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qict-figure-') as directory:
        rendered = Path(directory)/target.name
        figure.savefig(rendered, **kwargs)
        if target.suffix.lower() == '.pdf':
            data = rendered.read_bytes()
            if not data.startswith(b'%PDF-') or b'%%EOF' not in data[-2048:]:
                raise RuntimeError('Incomplete PDF render: '+target.name)
        descriptor, staging_name = tempfile.mkstemp(prefix=target.stem+'.', suffix='.staged', dir=target.parent)
        os.close(descriptor)
        staging = Path(staging_name)
        try:
            shutil.copyfile(rendered, staging)
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
