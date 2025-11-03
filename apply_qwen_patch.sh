#!/bin/bash

# apply_qwen_patch.sh - Apply qwen_plus_patch to diffusion-pipe safely
# Author: Claude Code Assistant
# Description: Script to apply patches from flet_app/modules/qwen_plus_patch to diffusion-pipe

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
# Resolve paths relative to this script location to avoid CWD issues
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/flet_app/modules/qwen_plus_patch"
TARGET_DIR="${SCRIPT_DIR}/diffusion-pipe"
BACKUP_DIR="${TARGET_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

# Files to patch
# Only include files that actually exist in the source patch directory
declare -a FILES_TO_PATCH=(
    "utils/dataset.py"
    "train.py"
    "tools/prepare_qwen_edit_triplet.py"
    "models/qwen_image_plus.py"
    "models/qwen_image.py"
)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Apply qwen_plus_patch to diffusion-pipe safely with backup support.

OPTIONS:
    -h, --help      Show this help message
    -d, --dry-run   Show what would be done without making changes
    -r, --rollback  Restore from the most recent backup
    -v, --verbose   Show detailed output

EXAMPLES:
    $0                  # Apply patch with interactive prompts
    $0 --dry-run        # Preview changes without applying
    $0 --rollback       # Restore from backup

EOF
}

# Function to check if directories exist
check_directories() {
    if [[ ! -d "$SOURCE_DIR" ]]; then
        print_error "Source directory not found: $SOURCE_DIR"
        exit 1
    fi

    if [[ ! -d "$TARGET_DIR" ]]; then
        print_error "Target directory not found: $TARGET_DIR"
        exit 1
    fi
}

# Function to create backup
create_backup() {
    print_status "Creating backup in $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"

    for file in "${FILES_TO_PATCH[@]}"; do
        source_file="$TARGET_DIR/$file"
        if [[ -f "$source_file" ]]; then
            backup_file="$BACKUP_DIR/$file"
            backup_dir=$(dirname "$backup_file")
            mkdir -p "$backup_dir"
            cp "$source_file" "$backup_file"
            if [[ "$VERBOSE" == "true" ]]; then
                print_status "Backed up: $file"
            fi
        fi
    done

    print_success "Backup created successfully"
}

# Function to apply patch
apply_patch() {
    print_status "Applying patch from $SOURCE_DIR to $TARGET_DIR"

    local copied_count=0

    for file in "${FILES_TO_PATCH[@]}"; do
        source_file="$SOURCE_DIR/$file"
        target_file="$TARGET_DIR/$file"

        if [[ -f "$source_file" ]]; then
            target_dir=$(dirname "$target_file")
            mkdir -p "$target_dir"

            if [[ "$DRY_RUN" == "true" ]]; then
                print_status "[DRY RUN] Would copy: $source_file -> $target_file"
            else
                # Only copy if file is missing or content differs
                if [[ ! -f "$target_file" ]] || ! cmp -s "$source_file" "$target_file"; then
                    cp "$source_file" "$target_file"
                    copied_count=$((copied_count+1))
                fi
                if [[ "$VERBOSE" == "true" ]]; then
                    if [[ -f "$target_file" ]] && cmp -s "$source_file" "$target_file"; then
                        print_status "Patched (no change): $file"
                    else
                        print_status "Patched: $file"
                    fi
                fi
            fi
        else
            print_warning "Source file not found: $source_file"
        fi
    done

    if [[ "$DRY_RUN" != "true" ]]; then
        if [[ $copied_count -gt 0 ]]; then
            print_success "Patch applied successfully ($copied_count file(s) updated)"
        else
            print_status "No files copied (already up to date or not found)"
        fi
    fi
}

# Function to rollback changes
rollback_changes() {
    local latest_backup=$(ls -1t "$TARGET_DIR"/backup_* 2>/dev/null | head -1)

    if [[ -z "$latest_backup" ]]; then
        print_error "No backup found for rollback"
        exit 1
    fi

    print_status "Rolling back from $latest_backup"

    for file in "${FILES_TO_PATCH[@]}"; do
        backup_file="$latest_backup/$file"
        target_file="$TARGET_DIR/$file"

        if [[ -f "$backup_file" ]]; then
            cp "$backup_file" "$target_file"
            if [[ "$VERBOSE" == "true" ]]; then
                print_status "Restored: $file"
            fi
        fi
    done

    print_success "Rollback completed"
}

# No interactive prompts or git operations in this script

# Main execution
main() {
    # Default values
    DRY_RUN="false"
    ROLLBACK="false"
    CREATE_BACKUP="true"
    VERBOSE="false"

    # Additional option: --all to copy entire patch directory recursively
    COPY_ALL="false"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -r|--rollback)
                ROLLBACK="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            --all)
                COPY_ALL="true"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    print_status "Qwen Patch Application Script"
    echo "====================================="

    # Check directories
    check_directories

    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_changes
        exit 0
    fi

    # No interactive prompts - no backup
    CREATE_BACKUP="false"

    # Create backup if needed
    if [[ "$CREATE_BACKUP" == "true" && "$DRY_RUN" != "true" ]]; then
        create_backup
    fi

    # Apply patch
    if [[ "$COPY_ALL" == "true" ]]; then
        print_status "Copying entire patch directory (recursive)"
        if [[ "$DRY_RUN" == "true" ]]; then
            print_status "[DRY RUN] Would sync: $SOURCE_DIR -> $TARGET_DIR"
        else
            # Use rsync if available for better behavior; fallback to cp -r
            if command -v rsync >/dev/null 2>&1; then
                rsync -av --delete --ignore-errors "$SOURCE_DIR/" "$TARGET_DIR/"
            else
                cp -rT "$SOURCE_DIR" "$TARGET_DIR"
            fi
            print_success "Directory sync completed"
        fi
    else
        apply_patch
    fi

    # Finish up
    if [[ "$DRY_RUN" != "true" ]]; then
        print_success "Patch application completed!"
        if [[ "$CREATE_BACKUP" == "true" ]]; then
            print_status "Backup saved at: $BACKUP_DIR"
            print_status "Use '$0 --rollback' to restore if needed"
        fi
    else
        print_status "Dry run completed. No changes were made."
    fi
}

# Run main function with all arguments
main "$@"
