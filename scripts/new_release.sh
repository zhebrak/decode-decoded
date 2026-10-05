#!/usr/bin/env bash
# Start a new engine release's configs from the previous one (a refresh):
#   scripts/new_release.sh vllm 0.30.0 0.31.0 vllm/vllm-openai:v0.31.0
# copies configs/engines/<engine>/<old>/arms.yaml to <new>/ with the version and image updated, pulls the image and
# records its digest and --help (scripts/capture_help.sh), and prints what to review: the flags in the new --help
# (renamed or removed ones break the arms), the arms' defaults (a mitigation may have become the default, as Model
# Runner V2 did in vLLM 0.29), and the golden files (UPDATE_GOLDEN=1 pytest tests/test_engines.py, then read the diff).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
engine=${1:?engine} old=${2:?old version} new=${3:?new version} image=${4:?image:tag}
src=configs/engines/$engine/$old dst=configs/engines/$engine/$new
[ -d "$src" ] || { echo "no $src" >&2; exit 1; }
[ -e "$dst" ] && { echo "$dst exists" >&2; exit 1; }
mkdir -p "$dst"
sed -e "s|^version: .*|version: $new|" -e "s|^image: .*|image: $image|" "$src/arms.yaml" > "$dst/arms.yaml"
sed -i "1s|^# .*|# $engine $new: started from $old by scripts/new_release.sh on $(date +%F); review every flag below against help.txt.|" "$dst/arms.yaml"
echo "$dst/arms.yaml written; now:"
echo "  1. a new configs/versions/<date>.yaml naming $engine: $new; in configs/grid/refresh.yaml point versions at it"
echo "     and list $engine under refresh.engines (session R is S2 again for the engines listed)"
echo "  2. GRID=configs/grid/refresh.yaml scripts/capture_help.sh $engine   (the image's digest and --help)"
echo "  3. diff $src/help.txt $dst/help.txt   (renamed or removed flags break the arms)"
echo "  4. add the release to tests/test_engines.py RELEASES; UPDATE_GOLDEN=1 pytest tests/test_engines.py; read the diff"
