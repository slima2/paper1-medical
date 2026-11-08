#!/usr/bin/env bash
# Environment for the encoder-swap extraction and the channel probes.
#
# The conda base already has a working CUDA torch 2.2.0 and timm 0.9.16, which
# is what the extraction needs and what several gigabytes of wheels would
# otherwise have to reproduce. The venv is therefore created with
# --system-site-packages and only the genuinely missing piece is installed on
# top: OpenSlide.
#
# openslide-bin ships the prebuilt C library as a wheel, so no apt or conda
# system package is required and nothing outside this venv is touched.
# openslide-python >= 1.4 finds it automatically.
#
#   bash scripts/setup_venv_dgx.sh
#   /home/rapids/notebooks/slima/venvs/slima_extract/bin/python scripts/...

set -euo pipefail

VENV=/home/rapids/notebooks/slima/venvs/slima_extract

echo "=== creating $VENV (inheriting conda base packages) ==="
python -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip

echo
echo "=== installing OpenSlide (binding + bundled C library) ==="
"$VENV/bin/pip" install --quiet "openslide-python>=1.4.0" openslide-bin

echo
echo "=== verification ==="
"$VENV/bin/python" -c "
import openslide, torch, timm, numpy, sklearn
from timm.layers import SwiGLUPacked
print('  openslide', openslide.__version__)
print('  torch    ', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.device_count(), 'gpu(s)')
print('  timm     ', timm.__version__, '| SwiGLUPacked present')
print('  numpy    ', numpy.__version__)
print('  sklearn  ', sklearn.__version__)
print()
print('  ready. use this interpreter:')
print('    $VENV/bin/python'.replace('\$VENV', '$VENV'))
"
