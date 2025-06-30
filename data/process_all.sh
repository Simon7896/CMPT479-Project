#!/bin/bash
set -e

TESTCASE_DIR="./C/testcases"
SUPPORT_DIR="./C/testcasesupport"
OUTPUT_DIR="./outputs"
TOOL_BIN="./build/juliet_tool"

mkdir -p "$OUTPUT_DIR"

find "$TESTCASE_DIR" -type f \( -name "*.c" -o -name "*.cpp" \) | while read -r FILE; do
    if [[ "$FILE" == *"__w32_"* ]]; then
        echo "Skipping Windows-specific file: $FILE"
        continue
    fi
    echo "Processing: $FILE"
    "$TOOL_BIN" "$FILE" -- -I"$SUPPORT_DIR"
done


echo "All files processed."

