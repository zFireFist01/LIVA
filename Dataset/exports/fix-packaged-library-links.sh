#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_ROOT="${1:-$PWD}"
PACKAGE_ROOT="$(cd -- "$PACKAGE_ROOT" && pwd)"
PACKAGE_NAME="$(basename -- "$PACKAGE_ROOT")"
LIBRARY_ROOT="$PACKAGE_ROOT/Dataset/builds/libraries"

case "$PACKAGE_NAME" in
    LIVA-PC2|LIVA-PC3|LIVA-PC4) ;;
    *)
        printf 'Expected LIVA-PC2, LIVA-PC3 or LIVA-PC4, found: %s\n' \
            "$PACKAGE_ROOT" >&2
        exit 2
        ;;
esac

[[ -d "$LIBRARY_ROOT" ]] || {
    printf 'Library directory not found: %s\n' "$LIBRARY_ROOT" >&2
    exit 2
}

fixed=0
while IFS= read -r -d '' link; do
    target="$(readlink -- "$link")"
    ln -sfn -- "${target#"$PACKAGE_NAME"/}" "$link"
    fixed=$((fixed + 1))
done < <(
    find "$LIBRARY_ROOT" -type l -lname "$PACKAGE_NAME/*" -print0
)

remaining="$(
    find "$LIBRARY_ROOT" -type l -lname "$PACKAGE_NAME/*" -print0 \
        | tr -cd '\0' \
        | wc -c
)"

if (( remaining != 0 )); then
    printf 'Repair incomplete: %s prefixed link(s) remain\n' "$remaining" >&2
    exit 1
fi

printf 'Library links repaired: %s\n' "$fixed"
