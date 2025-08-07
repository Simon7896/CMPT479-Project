#!/bin/bash
set -e

# Help message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [JOBS] [--skip-processed]"
    echo "       $0 [--skip-processed] [JOBS]"
    echo ""
    echo "Arguments:"
    echo "  JOBS             Number of parallel jobs (default: number of CPU cores)"
    echo "  --skip-processed Skip files that have already been processed (have output .json files)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Use all CPU cores, process all files"
    echo "  $0 4                    # Use 4 parallel jobs"
    echo "  $0 --skip-processed     # Skip already processed files, use all CPU cores"
    echo "  $0 4 --skip-processed   # Use 4 jobs and skip processed files"
    echo "  $0 --skip-processed 8   # Skip processed files and use 8 jobs"
    exit 0
fi

TESTCASE_DIR="./C/testcases"
SUPPORT_DIR="./C/testcasesupport"
OUTPUT_DIR="./outputs"
TOOL_BIN="./build/juliet_tool"

# Parse arguments
JOBS=${1:-$(nproc)}
SKIP_PROCESSED=${2:-false}

if [[ "$1" == "--skip-processed" ]]; then
    SKIP_PROCESSED=true
    JOBS=${2:-$(nproc)}
elif [[ "$2" == "--skip-processed" ]]; then
    SKIP_PROCESSED=true
fi

echo "Using $JOBS parallel jobs"
if [[ "$SKIP_PROCESSED" == "true" ]]; then
    echo "Skipping already processed files"
fi

mkdir -p "$OUTPUT_DIR"

# Function to process a single file
process_file() {
    local FILE="$1"
    local SUPPORT_DIR="$2"
    local TOOL_BIN="$3"
    local SKIP_PROCESSED="$4"
    local OUTPUT_DIR="$5"
    
    if [[ "$FILE" == *"__w32_"* ]]; then
        echo "Skipping Windows-specific file: $FILE"
        return 0
    fi
    
    # Check if file contains process.h header (Windows-specific)
    if grep -q "process\.h" "$FILE"; then
        echo "Skipping file with process.h header: $FILE"
        return 0
    fi
    
    # Check if already processed (if skip flag is enabled)
    if [[ "$SKIP_PROCESSED" == "true" ]]; then
        # Extract function name from file path and create expected output file name
        local BASENAME=$(basename "$FILE" .c)
        BASENAME=$(basename "$BASENAME" .cpp)
        local OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.json"
        
        if [[ -f "$OUTPUT_FILE" ]]; then
            echo "Skipping already processed file: $FILE (output exists: $OUTPUT_FILE)"
            return 0
        fi
    fi
    
    echo "Processing: $FILE"
    if "$TOOL_BIN" "$FILE" -- -I"$SUPPORT_DIR" >/dev/null 2>&1; then
        echo "✓ $(basename "$FILE")"
    else
        echo "✗ $(basename "$FILE") - FAILED"
        return 1
    fi
}

# Export function so parallel can use it
export -f process_file

# Find all files and process them in parallel
find "$TESTCASE_DIR" -type f \( -name "*.c" -o -name "*.cpp" \) | \
    parallel -j "$JOBS" process_file {} "$SUPPORT_DIR" "$TOOL_BIN" "$SKIP_PROCESSED" "$OUTPUT_DIR"

echo "All files processed."
