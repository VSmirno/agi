#!/bin/bash
# Remote experiment runner for mini-PC (AMD 92GB)
# Usage:
#   ./scripts/remote.sh sync          - sync code & rebuild image
#   ./scripts/remote.sh run <args>    - run scaling_test.py with args
#   ./scripts/remote.sh exec <cmd>    - run arbitrary python command
#   ./scripts/remote.sh logs          - show last run output
#   ./scripts/remote.sh shell         - interactive shell in container
#   ./scripts/remote.sh test          - run pytest

REMOTE_HOST="root@10.253.0.179"
REMOTE_PORT=2244
REMOTE_DIR="/opt/AGI"
IMAGE_NAME="snks-rocm"
SSH="ssh -p $REMOTE_PORT $REMOTE_HOST"

set -e

sync_code() {
    echo "==> Syncing code to $REMOTE_HOST:$REMOTE_DIR ..."
    # Create tar locally, pipe to remote
    tar -cf - \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='results' \
        --exclude='.pytest_cache' \
        -C "$(cd /d/Projects/AGI && pwd)" \
        src configs scripts tests Dockerfile \
    | $SSH "mkdir -p $REMOTE_DIR && tar -xf - -C $REMOTE_DIR"
    echo "==> Sync done"
}

build_image() {
    echo "==> Building Docker image on remote ..."
    $SSH "cd $REMOTE_DIR && docker build -t $IMAGE_NAME ."
    echo "==> Build done"
}

docker_run() {
    $SSH "docker run --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --shm-size=8g \
        -v $REMOTE_DIR/results:/app/results \
        $IMAGE_NAME \
        $@"
}

case "${1:-help}" in
    sync)
        sync_code
        build_image
        ;;
    run)
        shift
        docker_run scripts/scaling_test.py "$@"
        ;;
    exec)
        shift
        docker_run "$@"
        ;;
    test)
        docker_run -m pytest tests/ -x -q
        ;;
    logs)
        $SSH "docker logs \$(docker ps -lq) 2>&1" | tail -100
        ;;
    shell)
        $SSH -t "docker run --rm -it \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            --ipc=host \
            --shm-size=8g \
            -v $REMOTE_DIR/results:/app/results \
            --entrypoint bash \
            $IMAGE_NAME"
        ;;
    pull-results)
        mkdir -p results
        scp -P $REMOTE_PORT -r $REMOTE_HOST:$REMOTE_DIR/results/* results/
        echo "==> Results pulled to ./results/"
        ;;
    help|*)
        echo "Usage: $0 {sync|run|exec|test|logs|shell|pull-results}"
        echo ""
        echo "  sync              - sync code + rebuild Docker image"
        echo "  run [args]        - run scaling_test.py (e.g., run --sizes 50000,100000,200000)"
        echo "  exec <py-args>    - run arbitrary python3 command in container"
        echo "  test              - run pytest"
        echo "  logs              - show last container output"
        echo "  shell             - interactive bash in container"
        echo "  pull-results      - download results/ from remote"
        ;;
esac
