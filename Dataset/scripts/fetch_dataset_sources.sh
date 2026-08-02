#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$DATASET_DIR/manifests/source_manifest.json"
ONLY_KIND=""
ONLY_NAME=""
SKIP_ORIGIN_FILES=0
INCLUDE_DISABLED=0
FORCE=0
DOWNLOAD_ONLY=0
KEEP_TEMP=0
DRY_RUN=0
LIST_ONLY=0
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<'EOF'
Uso: fetch_dataset_sources.sh [opzioni]

Scarica/clona i sorgenti necessari in Dataset/sources, seguendo il manifest.
Non sposta i sorgenti dalla checkout attuale.

Opzioni:
  --manifest PATH          Manifest JSON (default: Dataset/manifests/source_manifest.json)
  --dataset-dir PATH       Root Dataset alternativa
  --only-kind KIND         Limita per tipo (`elf`, `library`,
                           `unseen-program`, `unseen-library`). Ripetibile.
  --only-name LIST         Limita ai nomi manifest separati da virgola
  --skip-origin-files      Usa solo le entry esplicite del manifest
  --include-disabled       Include entry disabilitate
  --force                  Rimuove destinazioni esistenti prima di scaricare
  --download-only          Scarica archivi senza estrarli
  --keep-temp              Mantiene directory temporanee dopo errori di estrazione
  --dry-run                Mostra le operazioni senza eseguirle
  --list                   Lista le entry espanse e termina
  -h, --help               Mostra questo messaggio

Esempi:
  Dataset/scripts/fetch_dataset_sources.sh --list
  Dataset/scripts/fetch_dataset_sources.sh --skip-origin-files
  Dataset/scripts/fetch_dataset_sources.sh --only-kind library --dry-run
EOF
}

die() {
    printf 'Errore: %s\n' "$*" >&2
    exit 1
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then
        "$@"
    fi
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "comando mancante: $1"
}

download_file() {
    local url="$1"
    local output="$2"

    if [[ -f "$output" ]]; then
        printf 'download skip existing %s\n' "$output"
        return
    fi

    printf 'download %s\n' "$url"
    if (( DRY_RUN )); then
        return
    fi

    mkdir -p -- "$(dirname -- "$output")"
    local tmp="$output.tmp"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -o "$tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp" "$url"
    else
        die "serve curl o wget per scaricare $url"
    fi
    mv -- "$tmp" "$output"
}

verify_sha256() {
    local file="$1"
    local expected="$2"

    [[ -z "$expected" || ! -f "$file" ]] && return
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] ||
        die "sha256 mismatch per $file: atteso $expected, ottenuto $actual"
}

archive_filename_from_url() {
    local url="$1"
    local path="${url%%\?*}"
    path="${path%%#*}"
    basename -- "$path"
}

extract_archive() {
    local archive="$1"
    local destination="$2"

    if [[ -e "$destination" ]]; then
        printf 'extract skip existing %s\n' "$destination"
        return
    fi

    printf 'extract %s -> %s\n' "$archive" "$destination"
    if (( DRY_RUN )); then
        return
    fi

    mkdir -p -- "$(dirname -- "$destination")"
    local temp_root
    temp_root="$(mktemp -d -p "$(dirname -- "$destination")" source-extract.XXXXXX)"

    if ! (
        case "$archive" in
            *.zip)
                need_cmd unzip
                unzip -q "$archive" -d "$temp_root"
                ;;
            *.tar|*.tar.gz|*.tgz|*.tar.xz|*.txz|*.tar.bz2|*.tbz2|*.tar.zst)
                tar -xf "$archive" -C "$temp_root"
                ;;
            *)
                die "formato archivio non supportato: $archive"
                ;;
        esac

        mapfile -t children < <(find "$temp_root" -mindepth 1 -maxdepth 1 ! -name '__MACOSX' -print)
        if ((${#children[@]} == 1)) && [[ -d "${children[0]}" ]]; then
            mv -- "${children[0]}" "$destination"
        else
            mkdir -p -- "$destination"
            shopt -s dotglob nullglob
            mv -- "$temp_root"/* "$destination"/
            shopt -u dotglob nullglob
        fi
    ); then
        if (( KEEP_TEMP )); then
            printf 'temporary extraction kept at %s\n' "$temp_root" >&2
        else
            rm -rf -- "$temp_root"
        fi
        return 1
    fi

    rm -rf -- "$temp_root"
}

clone_git() {
    local url="$1"
    local destination="$2"
    local revision="$3"

    if [[ -e "$destination" && "$FORCE" == 1 ]]; then
        run rm -rf -- "$destination"
    fi
    if [[ -e "$destination" ]]; then
        [[ -d "$destination/.git" ]] ||
            die "destinazione esistente non è un checkout git: $destination"
        if [[ -n "$revision" ]]; then
            local current_revision
            current_revision="$(git -C "$destination" rev-parse HEAD)"
            if [[ "$current_revision" != "$revision" ]]; then
                [[ -z "$(git -C "$destination" status --porcelain)" ]] ||
                    die "checkout con modifiche locali, impossibile fissare $revision: $destination"
                if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
                    run git -C "$destination" fetch origin "$revision"
                fi
                run git -C "$destination" checkout --detach "$revision"
            fi
            run git -C "$destination" submodule update --init --recursive
        fi
        printf 'git verified existing %s%s\n' \
            "$destination" "${revision:+ @ $revision}"
        return
    fi

    mkdir -p -- "$(dirname -- "$destination")"
    run git clone "$url" "$destination"
    if [[ -n "$revision" ]]; then
        run git -C "$destination" checkout --detach "$revision"
        run git -C "$destination" submodule update --init --recursive
    fi
}

fetch_archive() {
    local url="$1"
    local destination="$2"
    local sha256="$3"

    if [[ -e "$destination" && "$FORCE" == 1 ]]; then
        run rm -rf -- "$destination"
    fi

    local archive_name
    archive_name="$(archive_filename_from_url "$url")"
    local archive_path
    archive_path="$(dirname -- "$destination")/$archive_name"

    download_file "$url" "$archive_path"
    verify_sha256 "$archive_path" "$sha256"

    if (( ! DOWNLOAD_ONLY )); then
        extract_archive "$archive_path" "$destination"
    fi
}

parse_entries() {
    "$PYTHON" - "$MANIFEST" "$SKIP_ORIGIN_FILES" "$INCLUDE_DISABLED" "$ONLY_KIND" "$ONLY_NAME" <<'PY'
import csv
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
import sys

manifest_path = Path(sys.argv[1]).resolve()
skip_origin_files = sys.argv[2] == "1"
include_disabled = sys.argv[3] == "1"
only_kinds = {value for value in sys.argv[4].split(",") if value}
only_names = {value for value in sys.argv[5].split(",") if value}

archive_suffixes = (
    ".tar.gz", ".tar.xz", ".tar.bz2", ".tar.zst",
    ".tgz", ".tbz2", ".txz", ".zip",
)

def has_archive_suffix(url):
    filename = unquote(Path(urlparse(url).path).name)
    return any(filename.endswith(suffix) for suffix in archive_suffixes)

def is_git_url(url):
    parsed = urlparse(url)
    return (
        url.endswith(".git")
        or parsed.scheme in {"git", "ssh"}
        or url.startswith("git@")
        or (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.startswith("git.")
            and not has_archive_suffix(url)
        )
    )

def strip_archive_suffix(name):
    for suffix in archive_suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem

def archive_filename_from_url(url):
    name = unquote(Path(urlparse(url).path).name)
    if not name:
        raise ValueError(f"Cannot infer archive filename from URL: {url}")
    return name

def infer_archive_name(url):
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    filename = archive_filename_from_url(url)
    if "archive" in parts and "refs" in parts and "tags" in parts:
        archive_index = parts.index("archive")
        if archive_index > 0:
            repo = parts[archive_index - 1]
            tag = strip_archive_suffix(filename)
            return f"{repo}-{tag}"
    return strip_archive_suffix(filename)

def infer_git_name(url):
    if url.startswith("git@"):
        name = url.rsplit("/", maxsplit=1)[-1]
    else:
        name = Path(urlparse(url).path).name
    return name.removesuffix(".git")

def entry_from_url(kind, url, destination_root):
    url = url.strip()
    if not url or url.startswith("#"):
        return None
    if is_git_url(url):
        name = infer_git_name(url)
        source_type = "git"
    else:
        name = infer_archive_name(url)
        source_type = "archive"
    return {
        "kind": kind,
        "name": name,
        "type": source_type,
        "url": url,
        "destination": f"{destination_root.rstrip('/')}/{name}",
    }

def load_revisions(path):
    revisions = {}
    if not path.is_file():
        return revisions
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            revisions[(row["name"], row["url"])] = row["revision"]
    return revisions

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
entries = list(manifest.get("entries", []))

if not skip_origin_files:
    for origin in manifest.get("origin_files", []):
        origin_path = (manifest_path.parent / origin["path"]).resolve()
        if not origin_path.is_file():
            print(f"warn: origin file not found, skipping: {origin_path}", file=sys.stderr)
            continue
        revisions = {}
        if origin.get("git_revisions"):
            revisions = load_revisions((manifest_path.parent / origin["git_revisions"]).resolve())
        for line in origin_path.read_text(encoding="utf-8").splitlines():
            entry = entry_from_url(origin["kind"], line, origin["destination_root"])
            if entry is None:
                continue
            revision = revisions.get((entry["name"], entry["url"]))
            if revision:
                entry["revision"] = revision
            entries.append(entry)

seen = set()
for entry in entries:
    destination = entry.get("destination", "")
    if not destination or destination in seen:
        continue
    seen.add(destination)
    if not entry.get("enabled", True) and not include_disabled:
        continue
    if only_kinds and entry["kind"] not in only_kinds:
        continue
    if only_names and entry.get("name") not in only_names:
        continue
    values = [
        entry.get("kind", ""),
        entry.get("type", ""),
        "enabled" if entry.get("enabled", True) else "disabled",
        entry.get("name", ""),
        entry.get("url", ""),
        entry.get("destination", ""),
        entry.get("sha256", ""),
        entry.get("revision", ""),
        entry.get("note", ""),
    ]
    # A non-whitespace separator preserves empty fields in Bash `read`.
    print("\x1f".join(value.replace("\x1f", " ") for value in values))
PY
}

while (($#)); do
    case "$1" in
        --manifest)
            MANIFEST="$2"; shift 2 ;;
        --dataset-dir)
            DATASET_DIR="$(cd -- "$2" && pwd)"; shift 2 ;;
        --only-kind)
            [[ -n "$ONLY_KIND" ]] && ONLY_KIND+=","
            ONLY_KIND+="$2"; shift 2 ;;
        --only-name)
            [[ -n "$ONLY_NAME" ]] && ONLY_NAME+=","
            ONLY_NAME+="$2"; shift 2 ;;
        --skip-origin-files)
            SKIP_ORIGIN_FILES=1; shift ;;
        --include-disabled)
            INCLUDE_DISABLED=1; shift ;;
        --force)
            FORCE=1; shift ;;
        --download-only)
            DOWNLOAD_ONLY=1; shift ;;
        --keep-temp)
            KEEP_TEMP=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --list)
            LIST_ONLY=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            die "opzione sconosciuta: $1" ;;
    esac
done

SOURCES_DIR="$DATASET_DIR/sources"
MANIFEST="$(cd -- "$(dirname -- "$MANIFEST")" && pwd)/$(basename -- "$MANIFEST")"

need_cmd "$PYTHON"
need_cmd git
need_cmd tar
need_cmd sha256sum

failures=()
while IFS=$'\x1f' read -r kind source_type status name url relative_destination expected_sha256 revision note; do
    if (( LIST_ONLY )); then
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$kind" "$source_type" "$status" "$relative_destination" "$url"
        continue
    fi

    if [[ -z "$url" ]]; then
        printf 'skip %s: no URL (%s)\n' "$name" "${note:-no note}"
        continue
    fi

    destination="$SOURCES_DIR/$relative_destination"
    if [[ "$source_type" == "git" ]]; then
        if ! clone_git "$url" "$destination" "$revision"; then
            failures+=("$name")
        fi
    elif [[ "$source_type" == "archive" ]]; then
        if ! fetch_archive "$url" "$destination" "$expected_sha256"; then
            failures+=("$name")
        fi
    else
        printf 'tipo sorgente non supportato per %s: %s\n' "$name" "$source_type" >&2
        failures+=("$name")
    fi
done < <(parse_entries)

if (( LIST_ONLY )); then
    exit 0
fi

if ((${#failures[@]})); then
    printf '\nFailures:\n' >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

printf 'Done. Sources are under %s\n' "$SOURCES_DIR"
