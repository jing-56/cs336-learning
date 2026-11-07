#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
if [[ ! -x .venv/bin/python ]]; then
    python -m venv --system-site-packages .venv
fi
py=.venv/bin/python
if ! "$py" -c 'import torch; assert torch.cuda.is_available(); torch.zeros(1, device="cuda").cpu()'; then
    "$py" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
fi
if ! "$py" -c 'import numpy; assert int(numpy.__version__.split(".")[0]) >= 2'; then
    "$py" -m pip install 'numpy>=2,<3'
fi
if ! "$py" -c 'import regex'; then
    "$py" -m pip install regex
fi
"$py" run.py check
