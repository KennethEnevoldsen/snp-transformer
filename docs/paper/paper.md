

# Variant Transformer: GWAS using only Variants



# Abstract

# Introduction

- Multiple application seeks to represent the genetic predisposition of an individual 
- It is common to construct a polygenic risk score (PRS) from a set of genetic variants
  - A single value that represents the genetic predisposition of an individual
  - The PRS is constructed by summing the effect sizes of the variants weighted by the number of alleles
  - Problems: 
    - A single value does not capture the complexity of the genetic predisposition and does not allow for environmental interactions (i.e. a mean represent a bimoal distribution poorly)
    - It handles one disease at a time, 
      - even though some diseases are correlated and might provide and the predispisition of one disease might be correlated with the predisposition of another disease 
      - similarly they require large sample sizes, which for some diseases are not available and might never be available
- Promising Solution (learned vector representations):
  - Recent work NLP and vision processign have shown that semantic embedding of text sequences or image segment can obtain non-trivial few-shot performance, where a model pre-trained on a large corpus, can be further trained on a small dataset to obtain good performance
  - This is promising for genetic predisposition, as it allows for a model to be trained on a large dataset and then fine-tuned on a small dataset
  - Problems: These models are notably limited by their sequence length
    - Recent developments such as HyenaDNA have saught to improve the complexity to long genetic sequences better - however, they are still limited to sequence lengths of up to 1M nucleotides
    - scaling to long context is a ongoing problems in the field (longformer, retentive transformer, RWKW)
    - These models are notably intended for tasks such as regulatory element classification or chromatin profiling which does not require whole genome context
    - However, the genetic predisposition of an individual is a whole genome context problem 
- Our Solution:
  - We propose a transformer model that can represent the genetic predisposition of an individual using only minor alleles
    - In classical GWAS it is common to use SNPs as a proxy for the genetic predisposition of an individual, where you don't represent the whole genome, but only the SNPs
    - However these classical methods require a consistent input length and this the individual x SNP matrix is often sparsely populated (majority of SNP is only a variant for a small subset of the population)
    - This has limited classical methods to mainly linear model (although deep learning approaches exist (Arnor's paper))
    - Transformers are able to handle variable length input and thus can remove the sparse input problem by only representing the variants that are relevant for the given individual
    - This reduction allow for more complex modelling and thus a more complex representation of the genetic predisposition
  - For example, exome data contains approximately 20M SNPs, and so conventional methods would have to deal with matrices containing n x 20M elements, where n is the number of individuals. By contrast, a sparse encoding considers only elements with non-zero value. As the average frequence of exome SNPs is less than 5%, sparse encodes need only consider matrices with fewer than n x 1M elements.



# Methods


## Absolute vs relative positional Encodings

While the original transformer [vaswani_2017] model utilizes absolute positional encodings, multiple works
have since found relative positional encodings [alibi, rotary_embeddings] to be better for language modelling as the encoding of a word is relative to the encoding of the context. This is however not the case for our sparse encoding of base-pairs, which require absolute positional encoding to denote the global position of the SNP. 

# Results

# Discussion

# Conclusion

# References

# Acknowledgements

