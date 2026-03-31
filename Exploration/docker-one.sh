#!/usr/bin/env bash
set -euo pipefail

# One fixed image + one fixed container for this project.
IMAGE_NAME="thesis-binary-analysis:ubuntu22"
CONTAINER_NAME="thesis-binary-analysis-dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOUNT_DEST="/Exploration"

usage() {
  cat <<'EOF'
Usage:
  ./docker-one.sh build     Build/update the single project image
  ./docker-one.sh up        Create/start container with full Exploration mounted
  ./docker-one.sh shell     Open a shell in the single project container
  ./docker-one.sh stop      Stop the project container (if running)
  ./docker-one.sh clean     Remove dangling images and old tags of this project
  ./docker-one.sh rebuild   Rebuild image without cache

Optional env overrides:
  IMAGE_NAME=custom:tag CONTAINER_NAME=custom-name ./docker-one.sh build
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker not found in PATH." >&2
    exit 1
  fi
}

build_image() {
  echo "[build] Building ${IMAGE_NAME} from ${SCRIPT_DIR}/Dockerfile"
  docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
}

rebuild_image() {
  echo "[rebuild] Building ${IMAGE_NAME} with --no-cache"
  docker build --no-cache -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_has_expected_mount() {
  docker inspect "${CONTAINER_NAME}" \
    --format "{{range .Mounts}}{{if and (eq .Source \"${SCRIPT_DIR}\") (eq .Destination \"${MOUNT_DEST}\")}}OK{{end}}{{end}}" \
    | grep -Fxq "OK"
}

recreate_container() {
  if container_exists; then
    if container_running; then
      echo "[recreate] Stopping ${CONTAINER_NAME}"
      docker stop "${CONTAINER_NAME}" >/dev/null
    fi

    echo "[recreate] Removing ${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}" >/dev/null
  fi

  echo "[run] Creating container ${CONTAINER_NAME} from ${IMAGE_NAME}"
  docker run -dit \
    --name "${CONTAINER_NAME}" \
    -v "${SCRIPT_DIR}:${MOUNT_DEST}" \
    -w "${MOUNT_DEST}" \
    "${IMAGE_NAME}" \
    /bin/bash >/dev/null
}

ensure_container() {
  if ! container_exists; then
    recreate_container
  elif ! container_has_expected_mount; then
    echo "[recreate] Existing container does not match expected mount ${SCRIPT_DIR}:${MOUNT_DEST}"
    recreate_container
  fi

  if ! container_running; then
    echo "[start] Starting container ${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
}

up_container() {
  ensure_container
  echo "[up] Container ${CONTAINER_NAME} is ready with ${SCRIPT_DIR} mounted at ${MOUNT_DEST}"
}

open_shell() {
  ensure_container
  echo "[shell] Entering ${CONTAINER_NAME}"
  docker exec -it "${CONTAINER_NAME}" /bin/bash
}

stop_container() {
  if container_running; then
    echo "[stop] Stopping ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" >/dev/null
  else
    echo "[stop] Container ${CONTAINER_NAME} is not running"
  fi
}

clean_images() {
  echo "[clean] Removing dangling images"
  docker image prune -f >/dev/null || true

  echo "[clean] Keeping only ${IMAGE_NAME} tag for this project (if present)"
  # Remove other tags pointing to images for this project prefix.
  docker images --format '{{.Repository}}:{{.Tag}}' \
    | grep '^thesis-binary-analysis:' \
    | grep -v "^${IMAGE_NAME}$" \
    | xargs -r docker rmi || true
}

main() {
  require_docker

  cmd="${1:-}"
  case "${cmd}" in
    build)
      build_image
      ;;
    rebuild)
      rebuild_image
      ;;
    up)
      up_container
      ;;
    shell)
      open_shell
      ;;
    stop)
      stop_container
      ;;
    clean)
      clean_images
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
