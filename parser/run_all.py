"""
run_all.py — entry point: runs the parser then the validator.

Usage:
    python run_all.py
"""
import sys
import os
import time

PYTHON = r"C:\Users\LENOVO\AppData\Local\Programs\Python\Python39\python.exe"
WORKING_DIR = r"D:\DDI\drugbank_all_full_database.xml"


def run_step(label, script):
    print(f"\n{'='*65}")
    print(f" STEP: {label}")
    print(f"{'='*65}")
    t0 = time.time()
    ret = os.system(f'"{PYTHON}" "{os.path.join(WORKING_DIR, script)}"')
    elapsed = time.time() - t0
    status = "OK" if ret == 0 else f"FAILED (exit {ret})"
    print(f"\n[{label}] {status} in {elapsed:.1f}s")
    return ret == 0


if __name__ == "__main__":
    ok = run_step("Parse", os.path.join("parser", "main_parser.py"))
    if not ok:
        print("\n[run_all] Parser failed — skipping validation.")
        sys.exit(1)

    ok = run_step("Validate", os.path.join("parser", "validate.py"))
    sys.exit(0 if ok else 1)
