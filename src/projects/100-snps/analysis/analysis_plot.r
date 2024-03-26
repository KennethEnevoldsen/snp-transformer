library(ggplot2)
library(tidyverse)
library(ggdist)
library(patchwork)
# clear the environment


read_data <- function(split_train_path, split_test_path, y_hat_path) {
    # Splits
    split_train <- read_csv(split_train_path, col_names = FALSE) %>%
        rename(iid = X1) %>%
        # create new column for split
        mutate(split = "train")
    split_test <- read_csv(split_test_path, col_names = FALSE) %>%
        rename(iid = X1) %>%
        # create new column for split
        mutate(split = "test")
    split <- bind_rows(split_test, split_train) %>%
        mutate(split = as.factor(split))

    # Phenotypes
    y <- read_delim(y_path, col_names = FALSE, delim = " ") %>%
        rename(iid = X1, label = X3) %>%
        select(iid, label)

    # Predictions
    icd_str <- paste("icd", icd, sep = "")
    y_hat <- read_csv(y_hat_path) %>%
        rename(y_hat = !!icd_str) %>%
        select(iid, y_hat)

    # --- Merge the data ---
    data <- y %>%
        left_join(y_hat, by = "iid") %>%
        left_join(split, by = "iid")

    return(data)
}


plot_raincloud <- function(data, split, mdl_name_from_file, icd, mdl) { # nolint
    p <- data %>%
        filter(split == !!split) %>%
        dplyr::mutate(label = as.factor(label)) %>% # nolint
        ggplot(aes(x = label, y = y_hat, group = label)) + # nolint
        ## add half-violin from {ggdist} package
        ggdist::stat_halfeye(
            # ## custom bandwidth
            adjust = .5,
            # ## adjust height
            width = .6,
            # ## move geom to the right
            justification = -.1,
            # ## remove slab interval
            .width = 0,
            point_colour = NA,
            fill = "#9CAF88",
            alpha = 1,
        ) +
        geom_boxplot(
            width = .05,
            outlier.color = NA,
            fill = "#DFE6DA",
            alpha = 1,
        ) +
        ## add dot plots from {ggdist} package
        geom_point(
            ## draw horizontal lines instead of points
            shape = "|",
            size = 5,
            alpha = .1,
            color = "#758467",
            # move to the left
            position = position_nudge(x = -.08)
        ) +
        coord_flip() +
        labs(
            x = "True Label",
            y = "Prediction",
        ) +
        # ticks for y-axis
        scale_y_continuous(breaks = seq(0, 1, .1)) +
        theme_minimal() +
        theme(
            plot.background = element_rect(fill = "white"),
            legend.position = "none",
        ) +
        # make axis text larger
        theme(axis.text.x = element_text(size = 12), axis.text.y = element_text(size = 12)) +
        # set font
        theme(text = element_text(family = "Roboto Mono"))

    # add the information about the model to the plot
    mdl_name_from_file <- gsub(".csv", "", prediction)

    y_set <- 0.4
    x_set <- 0.7
    p <- p +
        annotate("text", parse = FALSE, x = 0.7 + x_set, y = y_set, label = paste("Split: ", split), hjust = 0) +
        annotate("text", parse = FALSE, x = 0.75 + x_set, y = y_set, label = paste("Model: ", mdl_name_from_file), hjust = 0) +
        annotate("text", parse = FALSE, x = 0.80 + x_set, y = y_set, label = paste("ICD: ", icd), hjust = 0) +
        annotate("text", parse = TRUE, x = 0.65 + x_set, y = y_set, label = paste("alpha == ", signif(summary(mdl)$coefficients[1, 1], 3)), hjust = 0) +
        annotate("text", parse = TRUE, x = 0.6 + x_set, y = y_set, label = paste("beta == ", signif(summary(mdl)$coefficients[2, 1], 3)), hjust = 0) +
        annotate("text", parse = FALSE, x = 0.55 + x_set, y = y_set, label = paste("p = ", signif(summary(mdl)$coefficients[2, 4], 3)), hjust = 0) +
        annotate("text", parse = TRUE, x = 0.5 + x_set, y = y_set, label = paste("R^2 == ", signif(summary(mdl)$r.squared, 3)), hjust = 0) +
        annotate("text", parse = TRUE, x = 0.45 + x_set, y = y_set, label = paste("n == ", nrow(data)), hjust = 0)
    return(p)
}

create_analysis_plot <- function(icd, prediction, wd_path) {
    y_path <- paste(wd_path, "/data/icd", icd, ".pheno", sep = "")
    y_hat_path <- paste(wd_path, "/data/predictions/", prediction, sep = "")
    split_test_path <- paste(wd_path, "data/test.split", sep = "/")
    split_train_path <- paste(wd_path, "data/train.split", sep = "/")
    mdl_name_from_file <- gsub(".csv", "", prediction)


    data <- read_data(split_train_path, split_test_path, y_hat_path)
    mdl_test <- lm(y_hat ~ label, data = data %>% filter(split == "test"))
    p_test <- plot_raincloud(data, "test", mdl_name_from_file, icd, mdl_test)
    mdl_train <- lm(y_hat ~ label, data = data %>% filter(split == "train"))
    p_train <- plot_raincloud(data, "train", mdl_name_from_file, icd, mdl_train)

    plot_name <- paste("data/predictions/", mdl_name_from_file, "_", icd, ".png", sep = "")
    ggsave(plot_name, p_test + p_train, width = 40, height = 20, units = "cm")
}


icd <- "511"
prediction <- "prediction_100snps-fine_tune_no_pretrain_multitask_big_batch_10k-warmup.csv"
project_path <- "/Users/au561649/Github/snp-transformer"
wd_path <- paste(project_path, "src/projects/100-snps/analysis", sep = "/")
create_analysis_plot(icd, prediction, wd_path)
