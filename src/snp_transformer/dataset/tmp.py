import pandas as pd

df = pd.read_csv(
    "/data-big-projects/snp-transformer/100_snps/code100_mhc_eur_full.sparse",
    sep=" ",
)

# remove nan rows
df = df.dropna()
