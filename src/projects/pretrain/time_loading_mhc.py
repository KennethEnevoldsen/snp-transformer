from pathlib import Path
import datetime 

from snp_transformer.dataset.dataset import IndividualsDataset

st = datetime.datetime.now()

home=Path.home()
data_folder = home / "snpher" / "faststorage"/"biobank"/"exomes"/"reformat"
data_path = data_folder / "mhc"
ds = IndividualsDataset(data_path)
et = datetime.datetime.now()
time_taken  = et - st

print(f"Time taken to load dataset: {time_taken.seconds//3600}:{(time_taken.seconds//60)%60}:{time_taken.seconds%60}")