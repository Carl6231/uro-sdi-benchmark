#!/usr/bin/env bash
# Run the Wu 2021 Python methods in separate processes. A CUDA failure on one
# section and method must not prevent the remaining methods from running.
set -u
cd "${URO_SDI_BASE:-$HOME/uro_sdi_bench}" || exit 1
export URO_SDI_BASE="$PWD"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
PY=${PY:-python}

shopt -s nullglob
slides=(data/processed/WU2021_*.h5ad)
for file in "${slides[@]}"; do
  slide=${file##*/}
  slide=${slide%.h5ad}
  args=(--slides "$slide" --seeds 0,1,2 --runs results/runs_ext)
  for method in leiden gmm smooth spagcn stagate graphst sedr spaceflow; do
    "$PY" scripts/08_run_methods.py "${args[@]}" --methods "$method" ||
      printf 'Process exited early on %s (%s)\n' "$slide" "$method" >&2
  done

  # An abrupt CUDA exit can leave no JSON. The runner skips successful seeds.
  for seed in 0 1 2; do
    if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' /dev/null \
        "results/runs_ext/$slide"/graphst_k*_s"$seed".json; then
      printf 'Retrying missing or failed GraphST seeds on CPU for %s\n' "$slide"
      CUDA_VISIBLE_DEVICES="" "$PY" scripts/08_run_methods.py "${args[@]}" --methods graphst ||
        printf 'GraphST CPU process exited early on %s\n' "$slide" >&2
      break
    fi
  done
done
