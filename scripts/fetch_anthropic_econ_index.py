from datasets import load_dataset
from pathlib import Path

out = Path("data/anthropic_econ_index/processed")
out.mkdir(parents=True, exist_ok=True)

# Load Anthropic Economic Index from Hugging Face
ds = load_dataset("Anthropic/EconomicIndex")

# Save each split as Parquet and CSV
for split, dset in ds.items():
    (out / f"anthropic_econ_index.{split}.parquet").parent.mkdir(parents=True, exist_ok=True)
    dset.to_parquet(out / f"anthropic_econ_index.{split}.parquet")
    dset.to_csv(out / f"anthropic_econ_index.{split}.csv", index=False)

print("Wrote:", sorted(p.name for p in out.glob("*")))
