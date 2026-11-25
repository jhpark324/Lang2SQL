#!/usr/bin/env bash
set -euo pipefail

transport=(--transport litellm:acompletion)
if [[ ${1:-} == --smoke ]]; then
    transport=()
    shift
fi
if [[ $# -ne 8 ]]; then
    echo 'Usage: matrix_ab.sh [--smoke] BASE_TREE BASE_SHA CANDIDATE_TREE CANDIDATE_SHA PREPARED SOURCE DB_DIR OUTPUT' >&2
    exit 2
fi

baseline_tree=$(realpath "$1")
baseline_sha=$2
candidate_tree=$(realpath "$3")
candidate_sha=$4
prepared=$(realpath "$5")
source=$(realpath "$6")
db_dir=$(realpath "$7")
output=$(realpath -m "$8")
gpt41_python=${BIRD_GPT41_PY:?Set BIRD_GPT41_PY to the absolute path of the gpt41 interpreter}
luna_python=${BIRD_LUNA_PY:?Set BIRD_LUNA_PY to the absolute path of the luna interpreter}

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p "$output/baseline" "$output/candidate"

run_lane() {
    local profile=$1 python=$2 condition=$3 arm=$4
    local product_tree=$baseline_tree product_sha=$baseline_sha
    if [[ $arm == candidate ]]; then
        product_tree=$candidate_tree
        product_sha=$candidate_sha
    fi
    local lane="$output/$arm/$condition-$profile"
    local resume=()
    if [[ -d "$lane" ]]; then
        resume=(--resume)
    fi

    printf 'Running %s/%s-%s\n' "$arm" "$condition" "$profile"
    "$python" -m bench.bird.evaluate run \
        --prepared "$prepared" --db-dir "$db_dir" --output "$lane" \
        --condition "$condition" --profile "$profile" \
        --product-tree "$product_tree" --product-commit "$product_sha" \
        "${transport[@]}" "${resume[@]}" > "$lane.run.log"
    "$python" -m bench.bird.evaluate score \
        --source "$source" --db-dir "$db_dir" --output "$lane" > "$lane.score.log"
    printf 'Scored %s/%s-%s\n' "$arm" "$condition" "$profile"
}

run_profile() (
    local profile=$1 python=$2 first=$3 second=$4
    run_lane "$profile" "$python" B "$first"
    run_lane "$profile" "$python" B "$second"
    run_lane "$profile" "$python" D "$second"
    run_lane "$profile" "$python" D "$first"
)

run_profile openai-gpt41-mini "$gpt41_python" baseline candidate &
mini_pid=$!
run_profile openai-gpt41 "$gpt41_python" candidate baseline &
full_pid=$!
run_profile openai-gpt6-luna "$luna_python" baseline candidate &
luna_pid=$!

result=0
for pid in "$mini_pid" "$full_pid" "$luna_pid"; do
    wait "$pid" || result=1
done
exit "$result"
