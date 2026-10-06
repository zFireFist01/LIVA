#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$DATASET_DIR/.." && pwd)"
REPO_PARENT="$(cd -- "$REPO_DIR/.." && pwd)"
IMAGE="${DATASET_IMAGE:-thesis-binary-datasets:2026-08-balanced}"

usage() {
    cat <<'EOF'
Uso: run_reproduction_container.sh [historical-libraries] [opzioni]

Costruisce l'ambiente Ubuntu 24.04 con GCC 11/13 e Clang 14/18, quindi esegue
una delle pipeline con la repo e la directory artefatti montate.

Senza azione, gli argomenti sono inoltrati a Dataset/create_datasets.sh.
Con `historical-libraries`, sono inoltrati a
Dataset/scripts/build_libraries.sh historical e nessun ELF viene modificato.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi

command -v docker >/dev/null 2>&1 || {
    printf 'Errore: docker non è disponibile\n' >&2
    exit 1
}

CONTAINER_COMMAND=(Dataset/create_datasets.sh "$@" --inside-container)
if [[ "${1:-}" == historical-libraries ]]; then
    shift
    CONTAINER_COMMAND=(Dataset/scripts/build_libraries.sh historical "$@")
fi

docker build --tag "$IMAGE" --file "$DATASET_DIR/Dockerfile" "$DATASET_DIR"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --env "THESIS_HOST_REPO_PARENT=$REPO_PARENT" \
    --env "THESIS_CONTAINER_REPO_PARENT=/workspace" \
    --volume "$REPO_PARENT:/workspace" \
    --workdir "/workspace/$(basename -- "$REPO_DIR")" \
    "$IMAGE" \
    "${CONTAINER_COMMAND[@]}"
