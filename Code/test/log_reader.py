
import os
from pathlib import Path

def read_utf16(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    try:
        with open(path, 'r', encoding='utf-16le') as f:
            content = f.read()
        print(f"--- {path} ---")
        # Find FAILED or ERROR blocks
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "FAILED" in line or "ERROR" in line:
                # Print 5 lines before and 10 lines after
                start = max(0, i - 2)
                end = min(len(lines), i + 15)
                print("\n".join(lines[start:end]))
                print("-" * 40)
    except Exception as e:
        print(f"Error reading {path}: {e}")

read_utf16("src/test/ml_results.txt")
read_utf16("src/test/dl_results.txt")
read_utf16("ml_results.txt")
read_utf16("dl_results.txt")
