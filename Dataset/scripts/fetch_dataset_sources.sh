#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$DATASET_DIR/manifests/source_manifest.json"
HISTORICAL_MANIFEST="$MANIFEST"
ONLY_KIND=""
ONLY_NAME=""
HISTORICAL_ONLY=""
SKIP_ORIGIN_FILES=0
INCLUDE_DISABLED=0
FORCE=0
DOWNLOAD_ONLY=0
KEEP_TEMP=0
DRY_RUN=0
LIST_ONLY=0
INCLUDE_HISTORICAL=0
HISTORICAL_JOBS=4
HISTORICAL_USE_COVERED=0
HISTORICAL_PRINT_BUILD_NAMES=0
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
  --historical             Include le versioni del manifest storico
  --historical-only LIST   Scarica solo le versioni storiche; LIST contiene
                           pacchetti separati da virgola (usa `all` per tutti)
  --historical-manifest P  Manifest storico alternativo
  --historical-jobs N      Download storici paralleli (default: 4)
  --use-covered            Per le historical usa una build simile già presente
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
  Dataset/scripts/fetch_dataset_sources.sh --historical-only all
  Dataset/scripts/fetch_dataset_sources.sh --historical-only acl,attr --historical-jobs 8
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
        --historical)
            INCLUDE_HISTORICAL=1; shift ;;
        --historical-only)
            (($# >= 2)) || die "$1 richiede un valore"
            INCLUDE_HISTORICAL=1
            HISTORICAL_ONLY="$2"
            shift 2 ;;
        --historical-manifest)
            (($# >= 2)) || die "$1 richiede un valore"
            HISTORICAL_MANIFEST="$2"; shift 2 ;;
        --historical-jobs)
            (($# >= 2)) || die "$1 richiede un valore"
            HISTORICAL_JOBS="$2"; shift 2 ;;
        --use-covered)
            HISTORICAL_USE_COVERED=1; shift ;;
        --historical-build-names)
            HISTORICAL_PRINT_BUILD_NAMES=1; shift ;;
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

[[ "$HISTORICAL_JOBS" =~ ^[1-9][0-9]*$ ]] ||
    die "--historical-jobs deve essere positivo"

historical_fetch_backend() {
    "$PYTHON" - "$@" <<'PY'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--only", default="")
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--use-covered", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--list", action="store_true")
parser.add_argument("--print-build-names", action="store_true")
args = parser.parse_args()

if args.jobs < 1:
    raise SystemExit("--historical-jobs deve essere positivo")

manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
packages = manifest.get("historical_packages", [])
selected = {item.strip().lower() for item in args.only.split(",") if item.strip()}
known = {str(package["name"]).lower() for package in packages}
unknown = selected - known
if unknown:
    raise SystemExit("Pacchetti historical sconosciuti: " + ", ".join(sorted(unknown)))

print_lock = threading.Lock()

def log(message, *, error=False):
    with print_lock:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)

def series(version):
    clean = version.split(":", 1)[-1]
    pieces = clean.split(".")
    return ".".join(pieces[:2]) if len(pieces) >= 2 else clean

entries = []
for package in packages:
    name = str(package["name"])
    if selected and name.lower() not in selected:
        continue
    for record in package["versions"]:
        version = str(record["version"])
        directory_version = str(record.get("directory_version", version)).replace(":", "_")
        context = {
            key: str(value)
            for source in (package, record)
            for key, value in source.items()
            if isinstance(value, (str, int, float))
        }
        context.setdefault("archive_version", version)
        context.setdefault("tag", version)
        context["series"] = series(context["archive_version"])
        context["directory_version"] = directory_version
        source_type = str(record.get("source_type", package["source_type"]))
        snapshot_package = str(
            record.get("snapshot_package", package.get("snapshot_package", name))
        )
        if source_type == "archive":
            url = str(record.get("url", package.get("url_template", ""))).format(**context)
            if not url:
                raise SystemExit(f"URL mancante per {name} {version}")
        elif source_type == "debian-snapshot":
            url = (
                "https://snapshot.debian.org/mr/package/"
                f"{snapshot_package}/{quote(version, safe='')}/srcfiles"
            )
        else:
            raise SystemExit(f"Tipo historical non supportato: {source_type}")
        covered_by = str(record.get("covered_by", ""))
        entries.append({
            "package": name,
            "label": str(record["label"]),
            "version": version,
            "directory": f"{name}-{directory_version}",
            "source_type": source_type,
            "snapshot_package": snapshot_package,
            "url": url,
            "covered_by": covered_by,
            "status": "covered" if covered_by and args.use_covered else "fetch",
        })

if args.print_build_names:
    seen = set()
    for entry in entries:
        directory = entry["covered_by"] if entry["status"] == "covered" else entry["directory"]
        if directory and directory not in seen:
            print(directory)
            seen.add(directory)
    raise SystemExit(0)

if args.list:
    for entry in entries:
        detail = f"covered by {entry['covered_by']}" if entry["status"] == "covered" else entry["url"]
        print(
            f"{entry['status']:<7}\t{entry['package']:<16}\t{entry['label']:<12}\t"
            f"{entry['directory']:<30}\t{detail}"
        )
    fetched = sum(entry["status"] == "fetch" for entry in entries)
    print(f"\nsource versions to fetch: {fetched}; covered by current builds: {len(entries)-fetched}")
    raise SystemExit(0)

sources = args.dataset_dir.resolve() / "sources" / "lib_sources"
cache = args.dataset_dir.resolve() / "sources" / "downloads" / "historical_libraries"
if not args.dry_run:
    sources.mkdir(parents=True, exist_ok=True)

def request_bytes(url, attempts=3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": "libseeker-source-fetch/1"}), timeout=120) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(attempt)
    raise last_error

def request_json(url):
    return json.loads(request_bytes(url).decode("utf-8"))

def download(url, destination):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(request_bytes(url))
    temporary.replace(destination)

def extract_archive(archive, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="historical-source-", dir=destination.parent) as temp_name:
        temporary = Path(temp_name)
        with tarfile.open(archive, mode="r:*") as source:
            source.extractall(temporary, filter="data")
        children = [child for child in temporary.iterdir() if child.name != "__MACOSX"]
        if len(children) == 1 and children[0].is_dir():
            children[0].replace(destination)
        else:
            destination.mkdir()
            for child in children:
                child.replace(destination / child.name)

def fetch_archive(entry):
    destination = sources / entry["directory"]
    if destination.exists():
        log(f"SKIP  {entry['directory']}: source directory already exists")
        return
    filename = Path(urlparse(entry["url"]).path).name
    archive = cache / "archives" / entry["directory"] / filename
    if args.dry_run:
        log(f"GET   {entry['directory']} <- {entry['url']}")
        return
    log(f"GET   {entry['directory']} <- {entry['url']}")
    download(entry["url"], archive)
    extract_archive(archive, destination)
    log(f"READY {destination}")

def snapshot_files(package, version):
    encoded = quote(version, safe="")
    listing = request_json(
        f"https://snapshot.debian.org/mr/package/{package}/{encoded}/srcfiles"
    )
    files = []
    for record in listing.get("result", []):
        digest = str(record["hash"])
        info = request_json(f"https://snapshot.debian.org/mr/file/{digest}/info")
        names = sorted({str(item["name"]) for item in info.get("result", [])})
        files.append((digest, names))
    if not any(name.endswith(".dsc") for _, names in files for name in names):
        raise RuntimeError(f"Nessun .dsc per {package} {version}")
    return files

def dsc_records(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1].isdigit() and len(fields[0]) in {32, 40, 64}:
            if all(character in "0123456789abcdefABCDEF" for character in fields[0]):
                records.append((fields[0].lower(), fields[2]))
    return records

def fetch_snapshot(entry):
    destination = sources / entry["directory"]
    if destination.exists():
        log(f"SKIP  {entry['directory']}: source directory already exists")
        return
    if args.dry_run:
        log(f"GET   {entry['directory']} <- {entry['url']}")
        return
    files = snapshot_files(entry["snapshot_package"], entry["version"])
    package_cache = cache / "debian" / entry["directory"]
    dsc_candidates = [(digest, name) for digest, names in files for name in names if name.endswith(".dsc")]
    suffix = "_" + entry["version"].split(":", 1)[-1] + ".dsc"
    dsc_digest, dsc_name = next((item for item in dsc_candidates if item[1].endswith(suffix)), dsc_candidates[0])
    dsc_path = package_cache / dsc_name
    log(f"GET   {entry['directory']} file={dsc_name}")
    download(f"https://snapshot.debian.org/file/{dsc_digest}", dsc_path)
    records = dsc_records(dsc_path)
    sha1_names = {digest: name for digest, name in records if len(digest) == 40}
    for digest, names in files:
        if digest == dsc_digest:
            continue
        name = sha1_names.get(digest.lower(), names[0])
        log(f"GET   {entry['directory']} file={name}")
        download(f"https://snapshot.debian.org/file/{digest}", package_cache / name)
    for _, required_name in records:
        required = package_cache / required_name
        if required.exists():
            continue
        name_suffix = required_name.partition("_")[2]
        aliases = [path for path in package_cache.iterdir() if path.is_file() and path.name.partition("_")[2] == name_suffix]
        if len(aliases) == 1:
            shutil.copyfile(aliases[0], required)
            log(f"ALIAS {entry['directory']} file={required_name} <- {aliases[0].name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="historical-debian-source-", dir=destination.parent) as temp_name:
        extracted = Path(temp_name) / "source"
        result = subprocess.run(
            ["dpkg-source", "--no-check", "-x", str(dsc_path), str(extracted)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError("dpkg-source failed: " + result.stdout.strip())
        extracted.replace(destination)
    log(f"READY {destination}")

def fetch(entry):
    if entry["source_type"] == "archive":
        fetch_archive(entry)
    else:
        fetch_snapshot(entry)

active = [entry for entry in entries if entry["status"] == "fetch"]
if not selected or "libxcb" in selected:
    active.append({
        "package": "xcb-proto",
        "label": "1.10-1",
        "version": "1.10-1",
        "directory": "xcb-proto-1.10",
        "source_type": "debian-snapshot",
        "snapshot_package": "xcb-proto",
        "url": "https://snapshot.debian.org/mr/package/xcb-proto/1.10-1/srcfiles",
        "covered_by": "",
        "status": "fetch",
    })
failures = []
with ThreadPoolExecutor(max_workers=min(args.jobs, max(1, len(active)))) as executor:
    futures = {executor.submit(fetch, entry): entry for entry in active}
    for future in as_completed(futures):
        entry = futures[future]
        try:
            future.result()
        except Exception as error:
            failures.append((entry["directory"], error))
            log(f"FAIL  {entry['directory']}: {error}", error=True)

if failures:
    log("\nFailed source downloads:", error=True)
    for directory, error in sorted(failures):
        log(f"  {directory}: {error}", error=True)
    raise SystemExit(1)
log(f"Done: {len(active)} historical source versions are available in {sources}")
PY
}

run_historical_fetch() {
    local args=(
        --manifest "$HISTORICAL_MANIFEST"
        --dataset-dir "$DATASET_DIR"
        --jobs "$HISTORICAL_JOBS"
    )
    if [[ -n "$HISTORICAL_ONLY" && "$HISTORICAL_ONLY" != all ]]; then
        args+=(--only "$HISTORICAL_ONLY")
    fi
    (( HISTORICAL_USE_COVERED )) && args+=(--use-covered)
    (( DRY_RUN )) && args+=(--dry-run)
    (( LIST_ONLY )) && args+=(--list)
    (( HISTORICAL_PRINT_BUILD_NAMES )) && args+=(--print-build-names)
    historical_fetch_backend "${args[@]}"
}

# `--historical-only` deliberately bypasses the normal source manifest.
if [[ -n "$HISTORICAL_ONLY" ]]; then
    run_historical_fetch
    exit $?
fi

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
    if (( INCLUDE_HISTORICAL )); then
        run_historical_fetch
    fi
    exit 0
fi

if ((${#failures[@]})); then
    printf '\nFailures:\n' >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

printf 'Done. Sources are under %s\n' "$SOURCES_DIR"

if (( INCLUDE_HISTORICAL )); then
    run_historical_fetch
fi
