#!/usr/bin/env bash
# Single entry point for this repo. Creates/uses .venv, then runs the task.
#
#   ./env.sh setup            venv + install package (editable) + sanity print
#   ./env.sh weights          download ng.pt (~2 GB) -> checkpoints/
#   ./env.sh dataset [SHARD]  download action labels -> data/nitrogen/
#   ./env.sh smoke [args]     end-to-end smoke test (add: --ckpt checkpoints/ng.pt --bf16)
#   ./env.sh test [args]      pytest (clones official repo into third_party/ for parity tests)
#   ./env.sh bench [args]     teacher latency / step-sweep benchmark -> results/
#   ./env.sh probe [args]     dataset label statistics -> results/
#   ./env.sh shell            interactive shell inside the venv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

ensure_venv() {
  if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -U pip -q
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}

case "${1:-setup}" in
  setup)
    ensure_venv
    pip install -e "$ROOT[dev]"
    python -c "import torch, ngd; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| ngd', ngd.__version__)"
    ;;
  weights)
    ensure_venv
    mkdir -p "$ROOT/checkpoints"
    hf download nvidia/NitroGen ng.pt --local-dir "$ROOT/checkpoints"
    ;;
  dataset)
    ensure_venv
    bash "$ROOT/scripts/download_dataset.sh" "${2:-SHARD_0000}"
    ;;
  smoke)
    ensure_venv
    python "$ROOT/scripts/smoke_test.py" "${@:2}"
    ;;
  test)
    ensure_venv
    if [ ! -d "$ROOT/third_party/NitroGen" ]; then
      git clone --depth 1 https://github.com/MineDojo/NitroGen.git "$ROOT/third_party/NitroGen"
    fi
    pytest "$ROOT/tests" -v "${@:2}"
    ;;
  bench)
    ensure_venv
    python "$ROOT/scripts/bench_teacher.py" "${@:2}"
    ;;
  probe)
    ensure_venv
    python "$ROOT/scripts/probe_labels.py" "${@:2}"
    ;;
  shell)
    ensure_venv
    exec "${SHELL:-bash}"
    ;;
  *)
    echo "usage: ./env.sh {setup|weights|dataset|smoke|test|bench|probe|shell}" >&2
    exit 1
    ;;
esac
