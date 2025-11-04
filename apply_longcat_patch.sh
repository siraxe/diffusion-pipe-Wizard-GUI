#!/bin/bash

# apply_longcat_patch.sh - Apply longcat_patch to diffusion-pipe safely
# Author: Claude Code Assistant
# Description: Script to apply patches from flet_app/modules/longcat_patch to diffusion-pipe

# set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
# Resolve paths relative to this script location to avoid CWD issues
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/flet_app/modules/longcat_patch"
TARGET_DIR="${SCRIPT_DIR}/diffusion-pipe"
BACKUP_DIR="${TARGET_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

# Files to patch
# Only include files that actually exist in the source patch directory
declare -a FILES_TO_PATCH=(
    ".gitmodules"
    "train.py"
    "examples/longcat.toml"
    "models/longcat_video.py"
)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
    return 0
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    return 0
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    return 0
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    return 0
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Apply longcat_patch to diffusion-pipe safely with backup support.

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

# Function to check submodule status
verify_submodule_status() {
    local submodule_path="$TARGET_DIR/submodules/LongCat-Video"
    local submodule_dir="$TARGET_DIR/submodules"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would check LongCat-Video submodule status"
        return
    fi

    print_status "Checking LongCat-Video submodule status..."

    # Check if .gitmodules exists and has LongCat-Video entry
    if [[ ! -f "$TARGET_DIR/.gitmodules" ]]; then
        print_error ".gitmodules file not found"
        return 1
    fi

    if ! grep -q "LongCat-Video" "$TARGET_DIR/.gitmodules"; then
        print_error "LongCat-Video submodule not found in .gitmodules"
        return 1
    fi

    # Check if submodule directory exists
    if [[ ! -d "$submodule_path" ]]; then
        print_warning "LongCat-Video submodule directory not found (needs initialization)"
        return 2
    fi

    # Check if submodule is properly initialized
    cd "$TARGET_DIR"
    local status_output=$(git submodule status submodules/LongCat-Video 2>/dev/null)
    if [[ -z "$status_output" ]]; then
        print_warning "LongCat-Video submodule not initialized"
        return 2
    fi

    # Parse status: first character indicates status
    local status_char=${status_output:0:1}
    case "$status_char" in
        " ")
            print_success "LongCat-Video submodule is up to date"
            return 0
            ;;
        "+")
            print_warning "LongCat-Video submodule has uncommitted changes"
            return 3
            ;;
        "-")
            print_warning "LongCat-Video submodule is not initialized"
            return 2
            ;;
        "U")
            print_error "LongCat-Video submodule has merge conflicts"
            return 4
            ;;
        *)
            print_warning "LongCat-Video submodule status: $status_output"
            return 5
            ;;
    esac
}

# Function to prompt user for submodule action
prompt_submodule_action() {
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would prompt for LongCat-Video submodule action"
        return 0
    fi

    local exit_code
    verify_submodule_status
    exit_code=$?

    echo ""
    print_status "=== LongCat-Video Submodule Management ==="

    case $exit_code in
        0)
            echo "✓ Submodule is up to date"
            echo ""
            read -p "Would you like to check for updates? (y/n): " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                return 2  # Update action
            else
                return 0  # Skip action
            fi
            ;;
        1)
            echo "✗ Configuration error (missing .gitmodules entry)"
            echo ""
            read -p "Would you like to skip submodule handling? (y/n): " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                return 1  # Skip action
            else
                return 9  # Abort
            fi
            ;;
        2)
            echo "✗ Submodule not initialized"
            echo "The LongCat-Video submodule is required for longcat model functionality."
            echo "Estimated download size: ~500MB - 2GB"
            echo "This may take several minutes depending on your network speed."
            echo ""
            read -p "Initialize LongCat-Video submodule? (y/n): " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                return 3  # Initialize action
            else
                return 1  # Skip action
            fi
            ;;
        3)
            echo "⚠ Submodule has uncommitted changes"
            echo ""
            read -p "Would you like to reset and update? (y/n): " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                return 2  # Update action (with reset)
            else
                return 0  # Skip action
            fi
            ;;
        4)
            echo "✗ Submodule has merge conflicts"
            echo ""
            read -p "Would you like to reset and reinitialize? (y/n): " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                return 4  # Reinitialize action
            else
                return 1  # Skip action
            fi
            ;;
        *)
            echo "? Unknown submodule status"
            echo ""
            read -p "Would you like to reinitialize submodule? (y/n): " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                return 4  # Reinitialize action
            else
                return 1  # Skip action
            fi
            ;;
    esac
}

# Ensure git knows about LongCat-Video before running git submodule commands
ensure_submodule_registered() {
    local submodule_name=$1
    local gitmodules_key="submodule.${submodule_name}.url"

    # Read URL from .gitmodules
    local submodule_url
    submodule_url=$(git config --file .gitmodules --get "$gitmodules_key" || true)

    if [[ -z "$submodule_url" ]]; then
        print_error "Submodule URL not found in .gitmodules for ${submodule_name}"
        return 1
    fi

    # Register in .git/config if missing
    if ! git config --get "$gitmodules_key" >/dev/null 2>&1; then
        print_status "Registering ${submodule_name} submodule with git"
        git config "$gitmodules_key" "$submodule_url"
        git config "submodule.${submodule_name}.active" true
        git config "submodule.${submodule_name}.update" checkout
        git submodule sync "$submodule_name" >/dev/null 2>&1 || git submodule sync >/dev/null 2>&1 || true
    elif [[ "$VERBOSE" == "true" ]]; then
        print_status "Submodule ${submodule_name} already registered with git"
    fi

    return 0
}

# Function to initialize/update submodule with retry logic
robust_submodule_operation() {
    local action=$1
    local max_retries=3
    local retry_count=0
    local submodule_name="submodules/LongCat-Video"

    if [[ "$DRY_RUN" == "true" ]]; then
        case $action in
            3) print_status "[DRY RUN] Would initialize LongCat-Video submodule" ;;
            2) print_status "[DRY RUN] Would update LongCat-Video submodule" ;;
            4) print_status "[DRY RUN] Would reinitialize LongCat-Video submodule" ;;
        esac
        return 0
    fi

    cd "$TARGET_DIR"

    # Ensure git knows about the submodule before running git submodule commands
    ensure_submodule_registered "$submodule_name" || return 1

    while [[ $retry_count -lt $max_retries ]]; do
        case $action in
            3)  # Initialize
                print_status "Initializing LongCat-Video submodule (attempt $((retry_count + 1))/$max_retries)..."
                if git submodule update --init --recursive "$submodule_name"; then
                    print_success "LongCat-Video submodule initialized successfully"; if [[ ! -f "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py" ]]; then touch "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py"; print_status "Created: longcat_video/__init__.py"; fi
                    return 0
                fi
                ;;
            2)  # Update
                print_status "Updating LongCat-Video submodule (attempt $((retry_count + 1))/$max_retries)..."
                if git submodule update --remote --merge "$submodule_name"; then
                    print_success "LongCat-Video submodule updated successfully"; if [[ ! -f "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py" ]]; then touch "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py"; print_status "Created: longcat_video/__init__.py"; fi
                    return 0
                fi
                ;;
            4)  # Reinitialize
                print_status "Reinitializing LongCat-Video submodule (attempt $((retry_count + 1))/$max_retries)..."
                git submodule deinit -f "$submodule_name" 2>/dev/null || true
                rm -rf "$TARGET_DIR/$submodule_name" 2>/dev/null || true
                if git submodule update --init --recursive "$submodule_name"; then
                    print_success "LongCat-Video submodule reinitialized successfully"; if [[ ! -f "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py" ]]; then touch "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py"; print_status "Created: longcat_video/__init__.py"; fi
                    return 0
                fi
                ;;
        esac

        retry_count=$((retry_count + 1))
        if [[ $retry_count -lt $max_retries ]]; then
            print_warning "Operation failed, retrying in 3 seconds..."
            sleep 3
        fi
    done

    print_error "Failed to complete submodule operation after $max_retries attempts"
    print_error "Please run manually:"
    case $action in
        3) print_error "cd $TARGET_DIR && git submodule update --init --recursive $submodule_name" ;;
        2) print_error "cd $TARGET_DIR && git submodule update --remote --merge $submodule_name" ;;
        4) print_error "cd $TARGET_DIR && git submodule deinit -f $submodule_name && git submodule update --init --recursive $submodule_name" ;;
    esac
    return 1
}

# Auto-initialize LongCat-Video submodule without prompts
auto_initialize_submodule() {
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would auto-initialize LongCat-Video submodule"
        return 0
    fi

    local exit_code
    verify_submodule_status
    exit_code=$? || true

    # Auto-handle based on status without prompting
    case $exit_code in
        0)
            if [[ "$VERBOSE" == "true" ]]; then
                print_status "LongCat-Video submodule is already up to date"
            fi
            return 0
            ;;
        1)
            print_error "Configuration error - LongCat-Video submodule not found in .gitmodules"
            return 1
            ;;
        2)
            print_status "Auto-initializing LongCat-Video submodule..."
            cd "$TARGET_DIR/submodules"
            if git clone https://github.com/meituan-longcat/LongCat-Video; then
                print_success "LongCat-Video submodule cloned successfully"; if [[ ! -f "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py" ]]; then touch "$TARGET_DIR/submodules/LongCat-Video/longcat_video/__init__.py"; print_status "Created: longcat_video/__init__.py"; fi
                return 0
            else
                # If clone fails because directory exists, check if it's already a valid git repo
                if [[ -d "LongCat-Video" && -d "LongCat-Video/.git" ]]; then
                    print_success "LongCat-Video submodule already exists and is valid"
                    return 0
                else
                    print_error "Failed to clone LongCat-Video submodule"
                    return 1
                fi
            fi
            ;;
        3)
            print_status "LongCat-Video submodule has uncommitted changes, updating..."
            if robust_submodule_operation 2; then
                print_success "LongCat-Video submodule updated successfully"
                return 0
            else
                print_error "Failed to update LongCat-Video submodule"
                return 1
            fi
            ;;
        4)
            print_status "LongCat-Video submodule has conflicts, reinitializing..."
            if robust_submodule_operation 4; then
                print_success "LongCat-Video submodule reinitialized successfully"
                return 0
            else
                print_error "Failed to reinitialize LongCat-Video submodule"
                return 1
            fi
            ;;
        *)
            print_status "Unknown submodule status, attempting initialization..."
            if robust_submodule_operation 3; then
                print_success "LongCat-Video submodule handled successfully"
                return 0
            else
                print_warning "Could not resolve LongCat-Video submodule status"
                return 1
            fi
            ;;
    esac
}

# Enhanced function to initialize git submodules
initialize_submodules() {
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would check and handle LongCat-Video submodule"
        return
    fi

    # Auto-initialize without prompts when running non-interactively
    auto_initialize_submodule
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

    # Initialize submodules after patching .gitmodules (for both dry run and actual run)
    if [[ " ${FILES_TO_PATCH[*]} " =~ " .gitmodules " ]]; then
        initialize_submodules
    fi

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

    print_status "Longcat Patch Application Script"
    echo "====================================="

    # Check directories
    check_directories

    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_changes
        exit 0
    fi

    # No interactive prompts - auto initialize submodule if needed
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
