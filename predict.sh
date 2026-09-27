#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    printf 'Usage: nnUNet_results=/path/to/results bash predict.sh {acdc|abdomenct1k|amos-ct} INPUT_DIR OUTPUT_DIR\n' >&2
    exit 2
fi
: "${nnUNet_results:?Set nnUNet_results to a writable directory}"

case "$1" in
    acdc)
        dataset=Dataset003_ACDC; id=3; tag=acdc-best-v1
        asset=SlimFormerPlusPlus_ACDC_best.zip
        digest=6aa31b16725fcb69ef366d873990a0cd44d96c47b974132a0df6b296025efef8 ;;
    abdomenct1k)
        dataset=Dataset002_AbdomenCT1K; id=2; tag=abdomenct1k-best-v1
        asset=SlimFormerPlusPlus_AbdomenCT1K_best.zip
        digest=10b13c86b3d4b03b99b2960e4a00e7ea85b457bdf8ea12e3356ac35c310ca33c ;;
    amos-ct)
        dataset=Dataset004_AMOS; id=4; tag=amos2022-ct-best-v1
        asset=SlimFormerPlusPlus_AMOS2022_CT_best.zip
        digest=ee72bd13a6d0d51a5eaed59621e6bc69a3896c76703a1dd813103d52396b3c3d ;;
    *) printf 'Unknown model: %s\n' "$1" >&2; exit 2 ;;
esac

input_dir=$2
output_dir=$3
if [[ ! -d "$input_dir" ]]; then
    printf 'Input directory does not exist: %s\n' "$input_dir" >&2
    exit 1
fi
for command in curl unzip sha256sum nnUNetv2_predict; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command is missing: %s\n' "$command" >&2
        exit 1
    fi
done
mkdir -p "$nnUNet_results"
model_dir="$nnUNet_results/$dataset/nnUNetTrainer_SlimFormerPlusPlus__nnUNetPlans__3d_fullres"
if [[ ! -e "$model_dir" ]]; then
    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT
    curl -fL --retry 3 -o "$temp_dir/$asset" \
        "https://github.com/WHHHHHY822/SlimFormerPlusPlus/releases/download/$tag/$asset"
    printf '%s  %s\n' "$digest" "$temp_dir/$asset" | sha256sum -c -
    unzip -q "$temp_dir/$asset" -d "$temp_dir/unpacked"
    (cd "$temp_dir/unpacked" && sha256sum -c SHA256SUMS)
    if [[ -d "$nnUNet_results/$dataset" ]]; then
        mv "$temp_dir/unpacked/$dataset/nnUNetTrainer_SlimFormerPlusPlus__nnUNetPlans__3d_fullres" \
            "$nnUNet_results/$dataset/"
    else
        mv "$temp_dir/unpacked/$dataset" "$nnUNet_results/"
    fi
    trap - EXIT
    rm -rf "$temp_dir"
fi
for required in plans.json dataset.json fold_0/checkpoint_best.pth; do
    if [[ ! -f "$model_dir/$required" ]]; then
        printf 'Incomplete model directory: %s\n' "$model_dir" >&2
        exit 1
    fi
done
export nnUNet_compile="${nnUNet_compile:-false}"
nnUNetv2_predict -i "$input_dir" -o "$output_dir" -d "$id" -c 3d_fullres \
    -tr nnUNetTrainer_SlimFormerPlusPlus -f 0 -chk checkpoint_best.pth
