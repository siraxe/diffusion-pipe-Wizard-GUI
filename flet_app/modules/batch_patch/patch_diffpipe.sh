#!/bin/bash

# patch_diffpipe.sh - Apply multiple patches to diffusion-pipe safely
# Author: Claude Code Assistant
# Description: Unified script to manage all diffusion-pipe patches with interactive selection

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/diffusion-pipe"
BACKUP_DIR="${TARGET_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

# Define patches with their metadata
declare -A PATCH_SOURCES=(
    ["longcat"]="${SCRIPT_DIR}/flet_app/modules/longcat_patch"
    ["qwen"]="${SCRIPT_DIR}/flet_app/modules/qwen_plus_patch"
    ["batch"]="${SCRIPT_DIR}/flet_app/modules/batch_patch"
)

declare -A PATCH_DESCRIPTIONS=(
    ["longcat"]="LongCat Video Model - Support for LongCat-Video model training"
    ["qwen"]="Qwen Plus Patch - Enhanced Qwen image model support"
    ["batch"]="Batch Size Change - Fix for resuming with different batch sizes"
)

declare -A PATCH_FILES=(
    ["longcat"]=".gitmodules|train.py|examples/longcat.toml|models/longcat_video.py"
    ["qwen"]="utils/dataset.py|train.py|tools/prepare_qwen_edit_triplet.py|models/qwen_image_plus.py|models/qwen_image.py"
    ["batch"]="utils/dataset.py|train.py"
)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
    echo ""
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Apply multiple patches to diffusion-pipe with interactive selection.

OPTIONS:
    -h, --help              Show this help message
    -d, --dry-run           Show what would be done without making changes
    -r, --rollback          Restore from the most recent backup
    -v, --verbose           Show detailed output
    -a, --all               Apply all patches without prompting
    --list                  List available patches
    --patch <name>          Apply specific patch (longcat, qwen, batch)

EXAMPLES:
    $0                      # Interactive patch selection
    $0 --all                # Apply all patches without prompting
    $0 --patch batch        # Apply only batch patch
    $0 --list               # Show available patches
    $0 --dry-run            # Preview changes without applying
    $0 --rollback           # Restore from backup

EOF
}

# Function to list available patches
list_patches() {
    print_header "Available Patches"

    for patch in "${!PATCH_DESCRIPTIONS[@]}"; do
        echo -e "${CYAN}$patch${NC}"
        echo "  Description: ${PATCH_DESCRIPTIONS[$patch]}"
        echo "  Source: ${PATCH_SOURCES[$patch]}"
        echo "  Files: ${PATCH_FILES[$patch]}"
        echo ""
    done
}

# Function to check if directories exist
check_directories() {
    if [[ ! -d "$TARGET_DIR" ]]; then
        print_error "Target directory not found: $TARGET_DIR"
        exit 1
    fi
}

# Function to check if patch source exists
patch_exists() {
    local patch=$1
    if [[ ! -d "${PATCH_SOURCES[$patch]}" ]]; then
        print_warning "Patch source not found: ${PATCH_SOURCES[$patch]}"
        return 1
    fi
    return 0
}

# Function to create backup
create_backup() {
    print_status "Creating backup in $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"

    for patch in "$@"; do
        IFS='|' read -ra files <<< "${PATCH_FILES[$patch]}"
        for file in "${files[@]}"; do
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
    done

    print_success "Backup created successfully at $BACKUP_DIR"
}

# Function to apply a single patch
apply_single_patch() {
    local patch=$1

    if ! patch_exists "$patch"; then
        print_error "Cannot apply patch '$patch' - source not found"
        return 1
    fi

    print_status "Applying $patch patch..."

    local source_dir="${PATCH_SOURCES[$patch]}"
    IFS='|' read -ra files <<< "${PATCH_FILES[$patch]}"
    local copied_count=0

    for file in "${files[@]}"; do
        source_file="$source_dir/$file"
        target_file="$TARGET_DIR/$file"

        if [[ -f "$source_file" ]]; then
            target_dir=$(dirname "$target_file")
            mkdir -p "$target_dir"

            if [[ "$DRY_RUN" == "true" ]]; then
                print_status "[DRY RUN] Would copy: $source_file -> $target_file"
            else
                if [[ ! -f "$target_file" ]] || ! cmp -s "$source_file" "$target_file"; then
                    cp "$source_file" "$target_file"
                    copied_count=$((copied_count+1))
                    print_status "Patched: $file"
                fi
            fi
        else
            print_warning "Source file not found: $source_file"
        fi
    done

    # Special handling for patches with submodules
    if [[ "$patch" == "longcat" ]] && [[ " ${files[*]} " =~ " .gitmodules " ]]; then
        if [[ "$DRY_RUN" != "true" ]]; then
            print_status "Initializing LongCat-Video submodule..."
            cd "$TARGET_DIR"
            if git submodule update --init --recursive submodules/LongCat-Video 2>/dev/null ||
               [[ -d "submodules/LongCat-Video" ]]; then
                print_success "LongCat-Video submodule ready"
            else
                print_warning "Could not initialize LongCat-Video submodule (may already be present)"
            fi
        fi
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        print_success "$patch patch applied ($copied_count file(s) updated)"
    fi

    return 0
}

# Function to apply patches
apply_patches() {
    local patches=("$@")

    if [[ ${#patches[@]} -eq 0 ]]; then
        print_error "No patches selected"
        return 1
    fi

    # Create backup for all selected patches
    if [[ "$DRY_RUN" != "true" ]] && [[ "$CREATE_BACKUP" == "true" ]]; then
        create_backup "${patches[@]}"
    fi

    # Apply each patch
    local failed=0
    for patch in "${patches[@]}"; do
        if ! apply_single_patch "$patch"; then
            failed=$((failed + 1))
        fi
    done

    if [[ $failed -eq 0 ]]; then
        print_success "All patches applied successfully!"
    else
        print_error "$failed patch(es) failed to apply"
        return 1
    fi

    return 0
}

# Function to prompt user for patch selection
interactive_patch_selection() {
    print_header "Patch Selection"

    local patches_to_apply=()

    for patch in "${!PATCH_DESCRIPTIONS[@]}"; do
        if ! patch_exists "$patch"; then
            print_warning "Patch '$patch' not found, skipping"
            continue
        fi

        echo ""
        echo -e "${CYAN}$patch${NC} - ${PATCH_DESCRIPTIONS[$patch]}"
        echo "Files to be patched: ${PATCH_FILES[$patch]}"
        read -p "Apply $patch patch? (y/n): " -r response

        if [[ "$response" =~ ^[Yy]$ ]]; then
            patches_to_apply+=("$patch")
            print_success "Selected: $patch"
        else
            print_status "Skipped: $patch"
        fi
    done

    if [[ ${#patches_to_apply[@]} -eq 0 ]]; then
        print_warning "No patches selected"
        return 1
    fi

    echo ""
    print_status "Selected patches: ${patches_to_apply[*]}"
    read -p "Proceed with applying selected patches? (y/n): " -r confirm

    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        apply_patches "${patches_to_apply[@]}"
        return $?
    else
        print_warning "Aborted by user"
        return 1
    fi
}

# Function to rollback changes
rollback_changes() {
    local latest_backup=$(ls -1t "$TARGET_DIR"/backup_* 2>/dev/null | head -1)

    if [[ -z "$latest_backup" ]]; then
        print_error "No backup found for rollback"
        exit 1
    fi

    print_status "Rollback from: $latest_backup"
    read -p "Confirm rollback? (y/n): " -r confirm

    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        print_warning "Rollback cancelled"
        return 1
    fi

    print_status "Rolling back..."

    # Get all patch names to find files to restore
    for patch in "${!PATCH_FILES[@]}"; do
        IFS='|' read -ra files <<< "${PATCH_FILES[$patch]}"
        for file in "${files[@]}"; do
            backup_file="$latest_backup/$file"
            target_file="$TARGET_DIR/$file"

            if [[ -f "$backup_file" ]]; then
                cp "$backup_file" "$target_file"
                if [[ "$VERBOSE" == "true" ]]; then
                    print_status "Restored: $file"
                else
                    print_success "Restored: $file"
                fi
            fi
        done
    done

    print_success "Rollback completed"
}

# Main execution
main() {
    # Default values
    DRY_RUN="false"
    ROLLBACK="false"
    CREATE_BACKUP="true"
    VERBOSE="false"
    APPLY_ALL="false"
    SPECIFIC_PATCH=""

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
            -a|--all)
                APPLY_ALL="true"
                shift
                ;;
            --list)
                list_patches
                exit 0
                ;;
            --patch)
                SPECIFIC_PATCH="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    print_header "Diffusion-Pipe Patch Manager"

    # Check directories
    check_directories

    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_changes
        exit $?
    fi

    # Handle specific patch
    if [[ -n "$SPECIFIC_PATCH" ]]; then
        if [[ -z "${PATCH_DESCRIPTIONS[$SPECIFIC_PATCH]}" ]]; then
            print_error "Unknown patch: $SPECIFIC_PATCH"
            list_patches
            exit 1
        fi
        apply_patches "$SPECIFIC_PATCH"
        exit $?
    fi

    # Handle apply all
    if [[ "$APPLY_ALL" == "true" ]]; then
        print_status "Applying all patches..."
        local all_patches=()
        for patch in "${!PATCH_DESCRIPTIONS[@]}"; do
            if patch_exists "$patch"; then
                all_patches+=("$patch")
            fi
        done
        apply_patches "${all_patches[@]}"
        exit $?
    fi

    # Interactive mode
    interactive_patch_selection
    local exit_code=$?

    # Show completion info
    if [[ $exit_code -eq 0 ]]; then
        if [[ "$DRY_RUN" != "true" ]]; then
            print_success "Patch application completed!"
            if [[ "$CREATE_BACKUP" == "true" ]]; then
                print_status "Backup saved at: $BACKUP_DIR"
                print_status "Use '$0 --rollback' to restore if needed"
            fi
        else
            print_status "Dry run completed. No changes were made."
        fi
    fi

    exit $exit_code
}

# Run main function with all arguments
main "$@"
