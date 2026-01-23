#!/bin/bash

# apply_patch.sh - Apply all patches to diffusion-pipe and LTX-2
# Description: Automatically applies all patches from patch/ directory

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/patch"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Print functions
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_header() { echo -e "\n${CYAN}==== $1 ====${NC}\n"; }

# Check if patch directory exists
if [[ ! -d "$PATCH_DIR" ]]; then
    print_error "Patch directory not found: $PATCH_DIR"
    exit 1
fi

# Function to apply patches for a specific target
apply_patches_for_target() {
    local patch_subdir=$1
    local target_dir=$2
    local target_name=$3

    local source_dir="${PATCH_DIR}/${patch_subdir}"

    # Skip if patch subdirectory doesn't exist
    if [[ ! -d "$source_dir" ]]; then
        return 0
    fi

    # Check if target directory exists
    if [[ ! -d "$target_dir" ]]; then
        print_error "Target directory not found: $target_dir"
        print_status "Make sure ${target_name} submodule is initialized"
        print_status "Run: git submodule update --init --recursive"
        return 1
    fi

    print_header "Applying Patches to ${target_name}"

    # Create backup
    local backup_dir="${target_dir}/backup_${TIMESTAMP}"
    print_status "Creating backup in $backup_dir"
    mkdir -p "$backup_dir"

    # Copy files, maintaining directory structure
    local files_copied=0
    while IFS= read -r -d '' source_file; do
        # Get relative path from patch subdirectory
        local rel_path="${source_file#$source_dir/}"
        local target_file="${target_dir}/${rel_path}"

        # Backup original file if it exists
        if [[ -f "$target_file" ]]; then
            local backup_file="${backup_dir}/${rel_path}"
            local backup_dirname=$(dirname "$backup_file")
            mkdir -p "$backup_dirname"
            cp "$target_file" "$backup_file"
        fi

        # Create target directory and copy file
        local target_dirname=$(dirname "$target_file")
        mkdir -p "$target_dirname"
        cp "$source_file" "$target_file"
        files_copied=$((files_copied + 1))
        print_status "Patched: ${target_name}/${rel_path}"

    done < <(find "$source_dir" -type f -print0)

    print_success "Patched $files_copied file(s) for ${target_name}"
    return 0
}

# Apply patches to diffusion-pipe
apply_patches_for_target "" "${SCRIPT_DIR}/diffusion-trainers/diffusion-pipe" "diffusion-pipe"

# Apply patches to LTX-2
apply_patches_for_target "LTX-2" "${SCRIPT_DIR}/diffusion-trainers/LTX-2" "LTX-2"

print_header "Complete"
print_success "All patches applied!"
print_status "Backups saved with timestamp: ${TIMESTAMP}"
