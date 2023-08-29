"""
!pip install pandas-plink
"""

from pathlib import Path

from datasets import Dataset

ds = Dataset.from_dict({"id": [1, 2], "snps": [[1, 2, 1], [2, 1, 2]]})

ds.save_to_disk("test_dataset")


from pandas_plink import read_plink1_bin

data_path = Path("tests/data/data.bed")


G = read_plink1_bin(
    str(data_path),
    str(data_path.with_suffix(".bim")),
    str(data_path.with_suffix(".fam")),
    verbose=False,
)


# read in using numpy

# !pip install bed-reader

import numpy as np
from bed_reader import open_bed

bed = open_bed(str(data_path))
val = bed.read(np.s_[1:, :3])
