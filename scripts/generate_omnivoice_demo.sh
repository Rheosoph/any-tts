#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

model="omnivoice"
speaker=""
speaker_label=""
speaker_description=""
language="german"
text_file="$repo_root/output/demo.txt"
cfg_scale="2.0"
output_path=""
features=""
profile="release"
default_omnivoice_features="${OMNIVOICE_FEATURES:-omnivoice,download,metal}"
default_qwen3_features="${QWEN3_TTS_FEATURES:-qwen3-tts,download,metal}"

normalize_model() {
        case "$1" in
                omnivoice)
                        printf 'omnivoice\n'
                        ;;
                qwen3|qwen3-tts)
                        printf 'qwen3\n'
                        ;;
                *)
                        echo "Unsupported model: $1" >&2
                        exit 1
                        ;;
        esac
}

usage() {
    cat <<'EOF'
Usage: scripts/generate_omnivoice_demo.sh [options]

Regenerate output/demo.txt with OmniVoice or Qwen3-TTS.

Options:
    --model NAME                    omnivoice or qwen3. Default: omnivoice
    --speaker NAME                  Qwen3 speaker name, or OmniVoice output label.
    --speaker-label LABEL           Explicit output label for the default file name.
    --speaker-description TEXT      Voice/style instruction string.
    --language CODE                 Language code or name. Default: german
  --text-file PATH                Input text file. Default: output/demo.txt
  --cfg-scale VALUE               Classifier-free guidance scale. Default: 2.0
    --output PATH                   Explicit output WAV path. Relative paths use the repo root.
    --features LIST                 Cargo feature list for the selected model.
  --debug                         Run without --release.
  --help                          Show this help.

Notes:
    OmniVoice does not support named speakers in this repo. For OmniVoice,
    --speaker is used as a file label and --speaker-description controls the voice.
    Qwen3-TTS uses --speaker as the named speaker selection and
    --speaker-description as the optional instruction.
EOF
}

sanitize_label() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_'
}

resolve_output_path() {
    local candidate="$1"
    if [[ "$candidate" == /* ]]; then
        printf '%s\n' "$candidate"
    else
        printf '%s\n' "$repo_root/$candidate"
    fi
}

prepare_output_path() {
    local candidate="$1"
    local resolved_path
    local output_dir

    resolved_path="$(resolve_output_path "$candidate")"
    output_dir="$(dirname "$resolved_path")"

    if ! mkdir -p "$output_dir"; then
        echo "Cannot create output directory: $output_dir" >&2
        if [[ "$candidate" == /* ]]; then
            echo "The path '$candidate' is absolute. Top-level directories like '/qwen3' are usually not writable on macOS." >&2
            echo "Use a repo-relative path such as 'output/qwen3_tts/flows_serena.wav' or a writable absolute path under \$HOME." >&2
        fi
        exit 1
    fi

    if [[ ! -w "$output_dir" ]]; then
        echo "Output directory is not writable: $output_dir" >&2
        echo "Choose a writable path under the repo or under \$HOME." >&2
        exit 1
    fi

    printf '%s\n' "$resolved_path"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$(normalize_model "$2")"
            shift 2
            ;;
        --speaker)
            speaker="$2"
            shift 2
            ;;
        --speaker-label)
            speaker_label="$2"
            shift 2
            ;;
        --speaker-description|--description|--instruct)
            speaker_description="$2"
            shift 2
            ;;
        --language)
            language="$2"
            shift 2
            ;;
        --text-file)
            text_file="$2"
            shift 2
            ;;
        --cfg-scale)
            cfg_scale="$2"
            shift 2
            ;;
        --output)
            output_path="$2"
            shift 2
            ;;
        --features)
            features="$2"
            shift 2
            ;;
        --debug)
            profile="debug"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$text_file" ]]; then
    echo "Text file not found: $text_file" >&2
    exit 1
fi

if [[ "$model" == "omnivoice" ]]; then
    if [[ -z "$speaker" ]]; then
        speaker="german_long"
    fi
    if [[ -z "$speaker_label" ]]; then
        speaker_label="$speaker"
    fi
    if [[ -z "$speaker_description" ]]; then
        speaker_description="female, calm delivery, neutral studio voice, standard German accent"
    fi
    if [[ -z "$features" ]]; then
        features="$default_omnivoice_features"
    fi
    example_name="generate_omnivoice"
    output_root="$repo_root/output/omnivoice"
    title="OmniVoice regeneration"
else
    if [[ -z "$speaker" ]]; then
        speaker="dylan"
    fi
    if [[ -z "$speaker_label" ]]; then
        speaker_label="$speaker"
    fi
    if [[ -z "$features" ]]; then
        features="$default_qwen3_features"
    fi
    example_name="generate_qwen3_tts"
    output_root="$repo_root/output/qwen3_tts"
    title="Qwen3-TTS regeneration"
fi

speaker_slug="$(sanitize_label "$speaker_label")"
speaker_slug="${speaker_slug#_}"
speaker_slug="${speaker_slug%_}"
if [[ -z "$speaker_slug" ]]; then
    speaker_slug="custom"
fi

if [[ -z "$output_path" ]]; then
    output_path="$output_root/demo_${speaker_slug}.wav"
fi

output_path="$(prepare_output_path "$output_path")"

cargo_args=(run --example "$example_name" --no-default-features --features "$features")
if [[ "$profile" == "release" ]]; then
    cargo_args+=(--release)
fi

echo "$title"
echo "  Model               : $model"
echo "  Text file           : $text_file"
echo "  Language            : $language"
echo "  CFG scale           : $cfg_scale"
echo "  Output              : $output_path"
echo "  Features            : $features"
if [[ "$model" == "omnivoice" ]]; then
    echo "  Speaker label       : $speaker_label"
    echo "  Speaker description : $speaker_description"
else
    echo "  Speaker             : $speaker"
    if [[ -n "$speaker_description" ]]; then
        echo "  Speaker description : $speaker_description"
    fi
fi

(
    cd "$repo_root"
    if [[ "$model" == "omnivoice" ]]; then
        OMNIVOICE_TEXT_FILE="$text_file" \
        OMNIVOICE_LANGUAGE="$language" \
        OMNIVOICE_INSTRUCT="$speaker_description" \
        OMNIVOICE_CFG_SCALE="$cfg_scale" \
        OMNIVOICE_OUTPUT="$output_path" \
        RUSTC_WRAPPER= \
        cargo "${cargo_args[@]}"
    else
        QWEN3_TTS_TEXT_FILE="$text_file" \
        QWEN3_TTS_LANGUAGE="$language" \
        QWEN3_TTS_SPEAKER="$speaker" \
        QWEN3_TTS_INSTRUCT="$speaker_description" \
        QWEN3_TTS_OUTPUT="$output_path" \
        RUSTC_WRAPPER= \
        cargo "${cargo_args[@]}"
    fi
)

echo "Wrote $output_path"
