library(ggplot2)
library(tidyverse)

setwd("src/projects/100-snps/analysis")


icd_code <- "499"
icd_path <- paste("data/icd", icd_code, ".pheno", sep = "")
sparse_path <- paste("data/code", icd_code, "_mhc_eur_full.sparse", sep = "")
fam_path <- paste("data/code", icd_code, "_mhc_eur_full.fam", sep = "")
split_test_path <- "data/test.split"
split_train_path <- "data/train.split"


# Splits
split_test <- read_csv("data/test.split", col_names = FALSE) %>%
    rename(iid = X1) %>%
    # create new column for split
    mutate(split = "test")
split_train <- read_csv("data/train.split", col_names = FALSE) %>%
    rename(iid = X1) %>%
    # create new column for split
    mutate(split = "train")
split <- bind_rows(split_test, split_train) %>%
    mutate(split = as.factor(split))


read_icd <- function(path) {
    read_delim(path, col_names = FALSE, delim = " ") %>%
        rename(iid = X1, label = X3) %>%
        select(iid, label)
}

# read in the data
icd_data <- icd_path %>%
    read_icd()

# read in the data
sparse <- read_delim(sparse_path, delim = " ") %>%
    select(ind = Individual, everything()) %>%
    # replace NAs with 0
    replace_na(list(Value = 0))
# ^ columns are ind, SNP, Value
# ind is row number in the .fam file

# read in the fam file
fam <- read_delim(fam_path, delim = " ", col_names = FALSE) %>%
    rename(iid = X1) %>%
    mutate(ind = row_number()) %>%
    select(ind, iid)

snps <- sparse %>%
    left_join(fam, by = "ind") %>% # add iid
    left_join(icd_data, by = "iid") %>% # add icd data
    left_join(split, by = "iid") # add split


# create a model for each split
train <- snps %>%
    filter(split == "train") %>%
    select(-split, -ind) %>%
    pivot_wider(names_from = SNP, values_from = Value)

test <- snps %>%
    filter(split == "test") %>%
    select(-split, -ind) %>%
    pivot_wider(names_from = SNP, values_from = Value)

# fit model on train

model <- lm(label ~ ., data = select(train, -iid))
summary(model)

# apply model
pred_train <- predict(model, newdata = select(train, -iid))
pred <- predict(model, newdata = select(test, -iid))

# create df
train_pred <- data.frame(iid = train$iid, pred = pred_train, label = train$label)
lm(label ~ pred, data = train_pred) %>% summary()

test_pred <- data.frame(iid = test$iid, pred = pred, label = test$label)
lm(label ~ pred, data = test_pred) %>% summary()


# # now with a elastic net
library(glmnet)
library(caret)

X <- train %>%
    select(-iid, -label) %>%
    as.matrix()

X_test <- test %>%
    select(-iid, -label) %>%
    as.matrix()

y <- train$label

y_test <- test$label

# create a grid of alpha and lambda
grid <- expand.grid(alpha = c(0.5, 0.3, 0.2, 1), lambda = c(0.1, 0.01, 0.001))

# fit model
model <- train(x = X, y = y, method = "glmnet", tuneGrid = grid, verbose = TRUE)
model


# predict
pred_train <- predict(model, newdata = X)
lm(label ~ pred, data = data.frame(pred = pred_train, label = y)) %>% summary()

pred_test <- predict(model, newdata = X_test)
lm(label ~ pred, data = data.frame(pred = pred_test, label = y_test)) %>% summary()
