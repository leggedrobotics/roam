#!/usr/bin/env bash
# Render the property-map examples with the hand-written labels in vlm_nav/example_data/labels.
# Needs the SAN server (vlm_nav/bin/run_san_server.sh). Pass LIVE=1 to ask the VLM instead
# (needs OPENAI_API_KEY or GEMINI_API_KEY; VLM_MODEL picks the model).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../vlm_nav"
out="$here/figures"
mkdir -p "$out"

run() {  # image setup temperature fields...
  local image=$1 setup=$2 temp=$3; shift 3
  local labels=()
  [[ "${LIVE:-0}" == 1 ]] || labels=(-l "example_data/labels/${image}_${setup}.json")
  local model=()
  [[ -n "${VLM_MODEL:-}" ]] && model=(--vlm_model "$VLM_MODEL")
  MPLBACKEND=Agg python3 example.py "example_data/${image}.jpg" -t "$setup" --seg_model api --save_images \
    --softmax_temperature "$temp" "${labels[@]}" "${model[@]}" > /dev/null
  python3 "$here/compose_figure.py" "result/${image}_${setup}" "$@" --out "$out/${image}_${setup}.png"
  echo "wrote $out/${image}_${setup}.png"
}

run snow_stairs_zurich terrain_properties 0.3 friction softness steppable
run stepping_stones_stream terrain_properties 0.3 friction softness steppable
run messy_desk tabletop_manipulation 0.05 graspable grasp_preference avoid friction fragility
run messy_desk object_interaction 0.05 movable pushable pullable mass
