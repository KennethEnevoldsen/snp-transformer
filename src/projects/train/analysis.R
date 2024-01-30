library(ggplot2)
library(tidyverse)

setwd("~/Downloads")

pred_file = "prediction_fine-tuned-no-pretrain[1].csv"

pheno = read_delim("icd511.pheno", col_names = FALSE, delim=" ") %>%
    rename(iid = X1, label = X3)  %>% 
    select(iid, label) %>% 
    mutate(label = as.factor(label))
pred = read_csv(pred_file) %>% rename(prob = icd511)
split_test = read_csv("test[1].split", col_names = FALSE) %>%
    rename(iid = X1)  %>% 
    # create new column for split
    mutate(split = "test")
split_train = read_csv("train[1].split", col_names = FALSE) %>%
    rename(iid = X1)  %>% 
    # create new column for split
    mutate(split = "train")

split = bind_rows(split_test, split_train) %>% 
    mutate(split = as.factor(split))


# merge data
merged = merge(pred, pheno, by = "iid") %>% 
    merge(split, by = "iid") %>%
    select(label, prob, split) %>%
    arrange(desc(label)) 

merged %>% head()

# plot boxplot
ggplot(merged, aes(group=label, x=label, y = prob)) + 
    geom_boxplot() + 
    labs(x = "Observed", y = "probability (ICD511=1)") +
    facet_grid(split ~ .)

# statistical test (is there a difference between the predicted probability of the two groups?)
lm(prob ~ 0+label, data = merged) %>% summary()

lm(prob ~ label, data = merged) %>% summary()
lm(prob ~ label*split, data = merged) %>% summary()


