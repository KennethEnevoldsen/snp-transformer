"""
minor preprocessing for pretraining for converting bim to details
"""
from pathlib import Path

from snp_transformer.dataset.convert import sparse_to_psparse
from snp_transformer.dataset.loaders import convert_bim_to_details, load_bim

home = Path.home()
data_folder = home / "snpher" / "faststorage" / "biobank" / "exomes" / "reformat"

# # Convert MHC bim to .details
# bim = load_bim(data_folder / "mhc.bim")
# details = convert_bim_to_details(bim)
# details.to_csv(data_folder / "mhc.details", sep=" ", index=False, header=False)


# convert MHC .sparse to .psparse
sparse_to_psparse(data_folder / "mhc.sparse", data_folder / "mhc.psparse", overwrite=True)



import polars as pl

df = pl.read_csv(data_folder / "mhc.sparse", separator=" ", 
                 dtypes={"Individual": pl.Utf8, "SNP": pl.Int64, "Value": pl.Int64})
test=df.partition_by("Individual", as_dict=True)
t = list(test.keys())
t[-1]

tdf = test[t[-1]]

tdf["SNP"]