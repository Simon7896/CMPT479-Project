#!/bin/bash
set -e

# Help message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [JOBS] [--skip-processed] [--progress]"
    echo "       $0 [--skip-processed] [JOBS] [--progress]"
    echo ""
    echo "Arguments:"
    echo "  JOBS             Number of parallel jobs (default: number of CPU cores)"
    echo "  --skip-processed Skip files that have already been processed (have output .json files)"
    echo "  --progress       Show detailed progress with filenames"
    echo ""
    echo "Examples:"
    echo "  $0                          # Use all CPU cores, process all files"
    echo "  $0 4                        # Use 4 parallel jobs"
    echo "  $0 --skip-processed         # Skip already processed files, use all CPU cores"
    echo "  $0 4 --skip-processed       # Use 4 jobs and skip processed files"
    echo "  $0 --progress               # Show detailed progress"
    exit 0
fi

TESTCASE_DIR="./C/testcases"
SUPPORT_DIR="./C/testcasesupport"
OUTPUT_DIR="./outputs"
TOOL_BIN="./build/juliet_tool"

# Parse arguments
JOBS=$(nproc)
SKIP_PROCESSED=false
SHOW_PROGRESS=false

for arg in "$@"; do
    case $arg in
        --skip-processed)
            SKIP_PROCESSED=true
            ;;
        --progress)
            SHOW_PROGRESS=true
            ;;
        [0-9]*)
            JOBS=$arg
            ;;
    esac
done

echo "Using $JOBS parallel jobs"
if [[ "$SKIP_PROCESSED" == "true" ]]; then
    echo "Skipping already processed files"
fi
if [[ "$SHOW_PROGRESS" == "true" ]]; then
    echo "Showing detailed progress"
fi

mkdir -p "$OUTPUT_DIR"

# Function to process a single file
process_file() {
    local FILE="$1"
    local SUPPORT_DIR="$2"
    local TOOL_BIN="$3"
    local SKIP_PROCESSED="$4"
    local OUTPUT_DIR="$5"
    local SHOW_PROGRESS="$6"
    
    if [[ "$FILE" == *"__w32_"* ]]; then
        if [[ "$SHOW_PROGRESS" == "true" ]]; then
            echo "⊘ $(basename "$FILE") - Windows-specific"
        fi
        return 0
    fi
    
    # Check if file contains process.h header (Windows-specific)
    if grep -q "process\.h" "$FILE" 2>/dev/null; then
        if [[ "$SHOW_PROGRESS" == "true" ]]; then
            echo "⊘ $(basename "$FILE") - process.h header"
        fi
        return 0
    fi
    
    # Check if already processed (if skip flag is enabled)
    if [[ "$SKIP_PROCESSED" == "true" ]]; then
        # Extract function name from file path and create expected output file name
        local BASENAME=$(basename "$FILE" .c)
        BASENAME=$(basename "$BASENAME" .cpp)
        local OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.json"
        
        if [[ -f "$OUTPUT_FILE" ]]; then
            if [[ "$SHOW_PROGRESS" == "true" ]]; then
                echo "↻ $(basename "$FILE") - already processed"
            fi
            return 0
        fi
    fi
    
    # Process the file
    if "$TOOL_BIN" "$FILE" -- -I"$SUPPORT_DIR" >/dev/null 2>&1; then
        if [[ "$SHOW_PROGRESS" == "true" ]]; then
            echo "✓ $(basename "$FILE")"
        fi
    else
        if [[ "$SHOW_PROGRESS" == "true" ]]; then
            echo "✗ $(basename "$FILE") - FAILED"
        fi
        return 1
    fi
}

# Export function so parallel can use it
export -f process_file

# Count total files first (for non-progress mode)
if [[ "$SHOW_PROGRESS" == "false" ]]; then
    echo "Counting files to process..."
    total_files=$(find "$TESTCASE_DIR" -type f \( -name "*.c" -o -name "*.cpp" \) | wc -l)
    echo "Found $total_files files to process"
    echo ""
fi

# Find all files and process them in parallel
if [[ "$SHOW_PROGRESS" == "true" ]]; then
    # Show individual file progress
    find "$TESTCASE_DIR" -type f \( -name "*.c" -o -name "*.cpp" \) | \
        parallel -j "$JOBS" --bar process_file {} "$SUPPORT_DIR" "$TOOL_BIN" "$SKIP_PROCESSED" "$OUTPUT_DIR" "$SHOW_PROGRESS"
else
    # Show overall progress bar
    find "$TESTCASE_DIR" -type f \( -name "*.c" -o -name "*.cpp" \) | \
        parallel -j "$JOBS" --bar process_file {} "$SUPPORT_DIR" "$TOOL_BIN" "$SKIP_PROCESSED" "$OUTPUT_DIR" "$SHOW_PROGRESS" 2>/dev/null
fi

echo ""
echo "All files processed."
