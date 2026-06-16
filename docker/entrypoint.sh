#!/usr/bin/env sh
set -eu

is_hydra_arg() {
    case "$1" in
        *=*|+*|-*) return 0 ;;
        *) return 1 ;;
    esac
}

has_model_override() {
    for arg in "$@"; do
        case "$arg" in
            model=*|+model=*|++model=*) return 0 ;;
        esac
    done
    return 1
}

has_demo_override() {
    for arg in "$@"; do
        case "$arg" in
            demo=*|+demo=*|++demo=*) return 0 ;;
        esac
    done
    return 1
}

has_checkpoint_override() {
    for arg in "$@"; do
        case "$arg" in
            demo.model_checkpoint=*|+demo.model_checkpoint=*|++demo.model_checkpoint=*) return 0 ;;
        esac
    done
    return 1
}

selected_model() {
    selected="${SECAI_DEMO_MODEL:-mobilevit_s}"
    for arg in "$@"; do
        case "$arg" in
            model=*|+model=*|++model=*) selected="${arg#*=}" ;;
        esac
    done
    printf '%s\n' "$selected"
}

run_default_demo() {
    if has_demo_override "$@"; then
        if has_model_override "$@"; then
            set -- python -m src.demo.qt_app "$@"
        else
            set -- python -m src.demo.qt_app "model=${SECAI_DEMO_MODEL:-mobilevit_s}" "$@"
        fi
    else
        if has_model_override "$@"; then
            set -- python -m src.demo.qt_app demo=default "$@"
        else
            set -- python -m src.demo.qt_app demo=default "model=${SECAI_DEMO_MODEL:-mobilevit_s}" "$@"
        fi
    fi
    prepare_models "$@"
    exec "$@"
}

prepare_models() {
    case "${SECAI_SKIP_MODEL_DOWNLOAD:-false}" in
        1|true|TRUE|yes|YES) return 0 ;;
    esac

    if has_checkpoint_override "$@"; then
        return 0
    fi

    if [ "$#" -ge 3 ] && [ "$1" = "python" ] && [ "$2" = "-m" ]; then
        case "$3" in
            src.demo.qt_app|src.demo.app)
                model_name="$(selected_model "$@")"
                python /app/docker/download_models.py --skip-if-present --required-model "$model_name"
                ;;
        esac
    fi
}

if [ "$#" -eq 0 ]; then
    run_default_demo
fi

if is_hydra_arg "$1"; then
    run_default_demo "$@"
fi

prepare_models "$@"
exec "$@"
