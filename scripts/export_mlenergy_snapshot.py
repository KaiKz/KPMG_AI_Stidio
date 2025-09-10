from pathlib import Path
import shutil, subprocess

# Adjust if you kept 'third_party' instead of 'external'
SRC = Path("external/mlenergy/data")
DST = Path("data/mlenergy/raw")
DST.mkdir(parents=True, exist_ok=True)

# Copy only common structured files
if SRC.exists():
    for p in SRC.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv",".json",".jsonl"}:
            rel = p.relative_to(SRC)
            (DST/rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, DST/rel)
else:
    raise SystemExit(f"Source not found: {SRC.resolve()}")

# Record the submodule commit for provenance
sha = subprocess.check_output(["git","-C","external/mlenergy","rev-parse","HEAD"]).decode().strip()
(Path("data/mlenergy/README.md")
 .write_text(f"Snapshot copied from external/mlenergy/data at commit {sha}\n"))

print("Copied files into", DST.resolve())
