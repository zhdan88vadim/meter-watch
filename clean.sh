#!/usr/bin/env bash
# clean.sh - Remove common build/cache artifacts from a project
#
# Usage:
#   ./clean.sh          # dry-run: show what would be removed
#   ./clean.sh --force  # actually delete
#   ./clean.sh -f       # same as --force

# Directories to remove (matched anywhere in the tree)
DIR_PATTERNS=(
    "__pycache__"
    ".pytest_cache"
    ".mypy_cache"
    ".ruff_cache"
    ".ipynb_checkpoints"
    ".tox"
    ".nox"
    ".eggs"
    "htmlcov"
    # "build"
    # "dist"
    ".hypothesis"
    ".pytype"
)

# File patterns to remove
FILE_PATTERNS=(
    "*.pyc"
    "*.pyo"
    "*.pyd"
    "*.so"
    ".coverage"
    ".coverage.*"
    "coverage.xml"
    "*.egg-info"
    ".DS_Store"
    "Thumbs.db"
)

FORCE=0
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

# Where to clean (default: current dir). Override via CLEAN_ROOT env var.
ROOT="${CLEAN_ROOT:-.}"

if [[ ! -d "$ROOT" ]]; then
    echo "Error: '$ROOT' is not a directory" >&2
    exit 1
fi

removed_dirs=0
removed_files=0

# Build -name arguments for find
dir_args=()
for p in "${DIR_PATTERNS[@]}"; do
    dir_args+=( -name "$p" -o )
done
unset 'dir_args[-1]'   # drop trailing -o

file_args=()
for p in "${FILE_PATTERNS[@]}"; do
    file_args+=( -name "$p" -o )
done
unset 'file_args[-1]'

# Remove directories (delete contents first; use -prune so find doesn't descend into them)
while IFS= read -r -d '' d; do
    if (( FORCE )); then
        rm -rf -- "$d"
        echo "removed dir : $d"
    else
        echo "would remove dir : $d"
    fi
    ((removed_dirs++)) || true
done < <(find "$ROOT" -type d \( "${dir_args[@]}" \) -prune -print0 2>/dev/null)

# Remove files
while IFS= read -r -d '' f; do
    if (( FORCE )); then
        rm -f -- "$f"
        echo "removed file: $f"
    else
        echo "would remove file: $f"
    fi
    ((removed_files++)) || true
done < <(find "$ROOT" -type f \( "${file_args[@]}" \) -print0 2>/dev/null)

echo
if (( FORCE )); then
    echo "Done. Removed $removed_dirs directories and $removed_files files."
else
    echo "Dry-run. Found $removed_dirs directories and $removed_files files."
    echo "Re-run with --force to delete."
fi