"""
minor preprocessing for pretraining for converting bim to details
"""
from pathlib import Path

from snp_transformer.dataset.loaders import convert_bim_to_details, load_bim

home = Path.home()
data_folder = home / "snpher" / "faststorage" / "biobank" / "exomes" / "reformat"

# Convert MHC bim to .details
bim = load_bim(data_folder / "mhc.bim")
details = convert_bim_to_details(bim)
details.to_csv(data_folder / "mhc.details", sep=" ", index=False, header=False)
