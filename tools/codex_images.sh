#!/bin/zsh
# Generate GPT-Image images with the Codex CLI, three at a time.
# Usage: tools/codex_images.sh tools/prompts/codex_train.txt data/codex_train
# Each prompt line is "name|prompt"; the image is saved as <out>/<name>.png (log in log_<name>.txt).
# Existing images are skipped, so an interrupted run can be resumed.
set -eu
prompts=${1:A} out=$2
mkdir -p "$out" && cd "$out"
one() {
  local line="$1" name="${1%%|*}" prompt="${1#*|}"
  [[ -f "$name.png" ]] && return 0
  codex exec --skip-git-repo-check -s workspace-write \
    "Use your image generation tool to create this image: $prompt. Save the generated image file into the current directory as $name.png. Do nothing else." \
    > "log_$name.txt" 2>&1
}
grep -v '^$' "$prompts" | tr '\n' '\0' | xargs -0 -P 3 -n 1 zsh -c "$(functions one); one \"\$1\"" _
