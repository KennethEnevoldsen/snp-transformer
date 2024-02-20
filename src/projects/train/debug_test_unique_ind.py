# see github issue for more

from pathlib import Path

import numpy as np
from snp_transformer.config.config_schemas import ResolvedConfigSchema
from snp_transformer.config.utils import load_config, parse_config

path = Path(
    "/home/kenneth/github/snp-transformer/src/projects/train/fine_tune_no_pretrain_only_511.cfg",
)

config_dict = load_config(path)
resolved_cfg: ResolvedConfigSchema = parse_config(config_dict)


dataset = resolved_cfg.train.training_dataset  # type: ignore

ds = iter(dataset)
individuals = [next(ds) for _ in range(30)]


# examine difference in the 10 first snps
for i in individuals:
    print(i.snps.bp[:10])


# generally repeated snps, but not exact duplicates

# Segments of some invitials seems to be contained within others
prev = None
for i in individuals:
    seg = np.array(i.snps.bp[:10])
    if prev is None:
        print("- first")
        prev = seg
        continue
    print(np.isin(seg, prev), end=" - all is same: ")
    # check all is the same
    print(np.all(np.isin(seg, prev)))
    prev = i.snps.bp[: 10 + 20]  # to account for injections


# There is def no exact duplicates:
ds = iter(dataset)
individuals = [next(ds) for _ in range(40)]


prev = set()
for i in individuals:
    seg = tuple(i.snps.bp[:10])
    if prev is None:
        print("- first")
        prev.add(seg)
        continue
    if seg in prev:
        print("in")

# output: no "in" printed


# What is the closest distance between any two individuals?
# I will use the first 10 snps for this


ds = iter(dataset)
individuals = [next(ds) for _ in range(39)]

snps = [i.snps.bp[:512] for i in individuals]


# define a custom distance function: Just wether A is in B
def similarity(x, y):  # noqa
    y_set = set(y)
    n_is_in = sum([1 for i in x if i in y_set])
    return n_is_in / len(x)


max_sim = 0
n_sim_above_0_5 = 0
n_sim_above_0_7 = 0
n_sim_above_0_8 = 0
n_sim_above_0_9 = 0
n_sim_above_1 = 0
for i in range(len(snps)):
    for j in range(i + 1, len(snps)):
        if i == j:
            continue
        sim = similarity(snps[i], snps[j])

        if sim > max_sim:
            max_sim = sim
            print(max_sim, i, j)

        if sim >= 0.5:
            n_sim_above_0_5 += 1
        if sim >= 0.7:
            n_sim_above_0_7 += 1
        if sim >= 0.8:
            n_sim_above_0_8 += 1
        if sim >= 0.9:
            n_sim_above_0_9 += 1
        if sim >= 1:
            n_sim_above_1 += 1

print(f"max_sim: {max_sim}")
# max_sim: 0.89 # noqa
print(
    f"n_sim_above_0_5: {n_sim_above_0_5}, n_sim_above_0_7: {n_sim_above_0_7}, n_sim_above_0_8: {n_sim_above_0_8}, n_sim_above_0_9: {n_sim_above_0_9}, n_sim_above_1: {n_sim_above_1}",
)
# n_sim_above_0_5: 203, n_sim_above_0_7: 29, n_sim_above_0_8: 3, n_sim_above_0_9: 0, n_sim_above_1: 0


# check that each person has a unique SNP
ds = iter(dataset)
individuals = [next(ds) for _ in range(40)]

snps = [i.snps.bp[:512] for i in individuals]

for i in range(len(snps)):
    all_snps_except_i = [s for j, s in enumerate(snps) if j != i]
    # unlist
    all_snps_except_i = {snp for snps_i in all_snps_except_i for snp in snps_i}

    unique_snps = [snp for snp in snps[i] if snp not in all_snps_except_i]

    print(
        f"Ind {i} has {len(snps[i])} snps, unique SNPs which no-one else has: {len(unique_snps)}",
    )

# Ind 0 has 512 snps, unique SNPs which no-one else has: 6
# Ind 1 has 512 snps, unique SNPs which no-one else has: 23
# Ind 2 has 512 snps, unique SNPs which no-one else has: 23
# Ind 3 has 512 snps, unique SNPs which no-one else has: 43
# Ind 4 has 512 snps, unique SNPs which no-one else has: 24
# Ind 5 has 512 snps, unique SNPs which no-one else has: 22
# Ind 6 has 512 snps, unique SNPs which no-one else has: 60
# Ind 7 has 512 snps, unique SNPs which no-one else has: 20
# Ind 8 has 512 snps, unique SNPs which no-one else has: 7
# Ind 9 has 512 snps, unique SNPs which no-one else has: 1
# Ind 10 has 512 snps, unique SNPs which no-one else has: 2
# Ind 11 has 512 snps, unique SNPs which no-one else has: 3
# Ind 12 has 431 snps, unique SNPs which no-one else has: 108
# Ind 13 has 512 snps, unique SNPs which no-one else has: 53
# Ind 14 has 512 snps, unique SNPs which no-one else has: 11
# Ind 15 has 512 snps, unique SNPs which no-one else has: 27
# Ind 16 has 512 snps, unique SNPs which no-one else has: 9
# Ind 17 has 512 snps, unique SNPs which no-one else has: 14
# Ind 18 has 512 snps, unique SNPs which no-one else has: 9
# Ind 19 has 512 snps, unique SNPs which no-one else has: 76
# Ind 20 has 512 snps, unique SNPs which no-one else has: 4
# Ind 21 has 512 snps, unique SNPs which no-one else has: 92
# Ind 22 has 512 snps, unique SNPs which no-one else has: 12
# Ind 23 has 512 snps, unique SNPs which no-one else has: 44
# Ind 24 has 512 snps, unique SNPs which no-one else has: 22
# Ind 25 has 512 snps, unique SNPs which no-one else has: 14
# Ind 26 has 512 snps, unique SNPs which no-one else has: 5
# Ind 27 has 512 snps, unique SNPs which no-one else has: 9
# Ind 28 has 512 snps, unique SNPs which no-one else has: 7
# Ind 29 has 512 snps, unique SNPs which no-one else has: 46
# Ind 30 has 512 snps, unique SNPs which no-one else has: 3
# Ind 31 has 512 snps, unique SNPs which no-one else has: 12
# Ind 32 has 512 snps, unique SNPs which no-one else has: 11
# Ind 33 has 512 snps, unique SNPs which no-one else has: 15
# Ind 34 has 512 snps, unique SNPs which no-one else has: 25
# Ind 35 has 512 snps, unique SNPs which no-one else has: 10
# Ind 36 has 512 snps, unique SNPs which no-one else has: 20
# Ind 37 has 512 snps, unique SNPs which no-one else has: 12
# Ind 38 has 512 snps, unique SNPs which no-one else has: 4
