“oal” - Occasional Active Learning
================
Andre Abadi
2024-10-28

## Introduction

The aim of this project is to develop a predictive model for binary
classification of legal documents. Many e-discovery platforms implement
such predictive models but are proprietary. This project aims to provide
an open source equivalent, not for production (necessarily) but for
educational purposes. We hope to meet or exceed the following metrics:

| Metric                   | Estimate |
|--------------------------|----------|
| Accuracy (50% threshold) | 0.74     |
| ROC-AUC                  | 0.77     |
| KAP                      | 0.46     |

### References

Our primary text reference for this project is *Supervised Machine
Learning for Text Analysis in R* (**SMLTAR**) by Emil Whitfield and
Julia Silge. This text is available as an interactive resource at
[smltar.com](https://smltar.com/).

### Style and Conventions

- Code is written using `snake_case` and attempts to follow the guidance
  of the online resource [*R for Data Science
  (2e)*](https://r4ds.hadley.nz/) by Wickham, Çetinkaya-Rundel, and
  Grolemund.
- We also adopt the [*Tidyverse Style
  Guide*](https://style.tidyverse.org/) with a preference toward piping
  operations where possible for code clarity.
- We use the native pipe where possible and fall back to the
  [*magrittr*](https://magrittr.tidyverse.org/) pipe where necessary.
- We also use the
  [*viridis*](https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html),
  colourising charts to improve accessibility of visualisations for
  readers with colour vision deficiency.
- We use [Tidymodels](https://www.tidymodels.org/) for our analysis, as
  it promotes good modelling practices by streamlining workflows,
  minimizing data leakage risks, and ensuring reproducibility.

## Import

    ## Rows: 5,000
    ## Columns: 8
    ## $ doc_id     <chr> "ENR.9000.0001.0040", "ENR.9000.0001.0133", "ENR.9000.0001.…
    ## $ relevant   <fct> NA, NA, NA, NA, 1, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, 0…
    ## $ date       <dttm> 2001-05-02 19:36:00, 2000-07-17 10:46:00, 2000-02-15 12:53…
    ## $ from       <chr> "phillip.allen@enron.com'", "phillip.allen@enron.com'", "ph…
    ## $ to         <chr> "james.steffes@enron.com'", "paul.lucci@enron.com'", "hunte…
    ## $ subject    <chr> NA, "Comments on Order 637 Compliance Filings", "Storage of…
    ## $ train_test <fct> NA, NA, NA, NA, train, test, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ content    <chr> "Jim, Is there going to be a conference call or some type o…

## Prepare

### Tidy

    ## Rows: 5,000
    ## Columns: 4
    ## $ doc_id     <chr> "ENR.9000.0001.0040", "ENR.9000.0001.0133", "ENR.9000.0001.…
    ## $ relevant   <fct> NA, NA, NA, NA, 1, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, 0…
    ## $ train_test <fct> NA, NA, NA, NA, train, test, NA, NA, NA, NA, NA, NA, NA, NA…
    ## $ content    <chr> "Jim, Is there going to be a conference call or some type o…

### Splits and Folds

### Preprocessing Recipe

## Modelling

### Null Model

> “a ‘null model’ or baseline model, \[is\] a simple, non-informative
> model that always predicts the largest class for classification. Such
> a model is perhaps the simplest heuristic or rule-based alternative
> that we can consider as we assess our modelling efforts.”

    ##    .metric      mean    std_err
    ## 1 accuracy 0.4883402 0.01210339
    ## 2  roc_auc 0.5000000 0.00000000

    ##    .metric      mean    std_err
    ## 1 accuracy 0.7193495 0.01505729
    ## 2  roc_auc 0.7693570 0.01255672
