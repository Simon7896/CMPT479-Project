#!/usr/bin/env python3
"""
Multithreaded C/C++ file processor using Python multiprocessing
"""

import os
import subprocess
import multiprocessing as mp
from pathlib import Path
import argparse
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def should_skip_file(file_path: str) -> tuple[bool, str]:
    """Check if file should be skipped and return reason"""
    
    # Skip Windows-specific files
    if "__w32_" in str(file_path):
        return True, "Windows-specific file"
    
    # Check for process.h header
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'process.h' in content:
                return True, "Contains process.h header"
    except Exception as e:
        return True, f"Error reading file: {e}"
    
    return False, ""

def process_single_file(args: tuple[str, str, str]) -> tuple[bool, str, str]:
    """Process a single file"""
    file_path, support_dir, tool_bin = args
    
    # Check if file should be skipped
    should_skip, skip_reason = should_skip_file(file_path)
    if should_skip:
        return True, file_path, f"Skipped: {skip_reason}"
    
    try:
        # Run the tool
        cmd = [tool_bin, file_path, "--", f"-I{support_dir}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, file_path, "Processed successfully"
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to process: {e.stderr[:200]}..."
        return False, file_path, error_msg
    except Exception as e:
        return False, file_path, f"Error: {str(e)}"

def find_source_files(testcase_dir: str) -> List[str]:
    """Find all C/C++ source files"""
    source_files = []
    testcase_path = Path(testcase_dir)
    
    for pattern in ["**/*.c", "**/*.cpp"]:
        source_files.extend(testcase_path.glob(pattern))
    
    return [str(f) for f in source_files]

def process_files_parallel(testcase_dir: str, support_dir: str, tool_bin: str, num_workers: int = None):
    """Process files in parallel using multiprocessing"""
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    logger.info(f"Using {num_workers} worker processes")
    
    # Find all source files
    logger.info("Finding source files...")
    source_files = find_source_files(testcase_dir)
    logger.info(f"Found {len(source_files)} source files")
    
    # Prepare arguments for each file
    file_args = [(file_path, support_dir, tool_bin) for file_path in source_files]
    
    # Process files in parallel
    successful = 0
    failed = 0
    skipped = 0
    
    with mp.Pool(processes=num_workers) as pool:
        logger.info("Starting parallel processing...")
        
        # Use imap for better progress tracking
        for i, (success, file_path, message) in enumerate(pool.imap(process_single_file, file_args)):
            if success:
                if "Skipped" in message:
                    skipped += 1
                    logger.debug(f"{file_path}: {message}")
                else:
                    successful += 1
                    logger.info(f"[{i+1}/{len(file_args)}] {file_path}: {message}")
            else:
                failed += 1
                logger.error(f"[{i+1}/{len(file_args)}] {file_path}: {message}")
            
            # Progress update every 100 files
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{len(file_args)} files processed")
    
    logger.info(f"Processing complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Total: {len(file_args)}")

def main():
    parser = argparse.ArgumentParser(description="Process C/C++ files in parallel")
    parser.add_argument("--testcase-dir", default="./C/testcases", help="Directory containing test cases")
    parser.add_argument("--support-dir", default="./C/testcasesupport", help="Support directory for includes")
    parser.add_argument("--tool-bin", default="./build/juliet_tool", help="Path to processing tool")
    parser.add_argument("--workers", type=int, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify tool exists
    if not os.path.exists(args.tool_bin):
        logger.error(f"Tool not found: {args.tool_bin}")
        return 1
    
    # Verify directories exist
    if not os.path.exists(args.testcase_dir):
        logger.error(f"Testcase directory not found: {args.testcase_dir}")
        return 1
    
    if not os.path.exists(args.support_dir):
        logger.error(f"Support directory not found: {args.support_dir}")
        return 1
    
    try:
        process_files_parallel(
            args.testcase_dir, 
            args.support_dir, 
            args.tool_bin, 
            args.workers
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
