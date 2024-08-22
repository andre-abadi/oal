“oal” - Occasional Active Learning
================
Andre Abadi
2024-08-22

## Libraries

``` r
library(tidyverse) # read_csv, mutate
library(tidytext) # unnest_tokens
library(textclean) # replace_contraction
require(stringi) # stri_trans_ncf
require(stringr) # str_squish
library(torch)
library(data.table) # fread
```

## Import

``` r
require(data.table) # fread
require(tidyverse) # as_tibble
enron <-
  fread(
    "data/enron_edisco.csv", # file to read
    select = c("doc_id", "relevant", "content") # only these columns
  ) |> 
  as_tibble() |> # tidyverse
  filter(!is.na(relevant)) # drop rows where relevant = NA
# excerpt
enron$content[23] |>
  str_wrap(width = 75) |>
  cat()
```

    ## Please review the attachment and let me know if you have any changes.
    ## The final document will provide a starting place for the year end review
    ## process.

## Clean

``` r
require(tidyverse) # mutate
require(tidytext) # stop_words
require(textclean) # replace_contraction
require(stringi) # stri_trans_ncf
require(stringr) # str_squish
stop_words_pattern <- 
  paste0("\\b(", paste(stop_words$word, collapse = "|"), ")\\b")
regex_patterns <- 
  list(
    "[^a-zA-Z]",  # non-words
    "https?://\\S+|www\\.\\S+",  # URLs
    "[\\w._%+-]+@[\\w.-]+\\.[a-zA-Z]{2,}",  # emails
    "[\\w\\-]+(/|\\\\)[\\w\\-]+(/|\\\\)?"  # file paths
)
combined_regex <- 
  paste0("(", paste0(regex_patterns, collapse = "|"), ")")
enron <-
  enron |>
  mutate(
    content = str_replace_all(content, combined_regex, " "), # apply regex
    content = tolower(content), # to lower case
    content = replace_contraction(content),
    content = stri_trans_nfc(content), # unicode normalisation
    content = str_replace_all(content, stop_words_pattern, ""),
    content = str_squish(content) # max 1 whitespace, remove start and end
  ) |>
  filter(content != "" & !is.na(content)) # remove now-empty content rows
rm(
  regex_patterns,
  combined_regex,
  stop_words_pattern
)
# excerpt
enron$content[23] |>
  str_wrap(width = 75) |>
  cat()
```

    ## review attachment final document provide starting review process

### Message Lengths

``` r
require(ggplot2)
enron %>%
  mutate(content_length = nchar(content)) %>%
  ggplot(aes(x = content_length, fill = ..count..)) +
  geom_histogram(binwidth = 0.5,) +
  #scale_x_log10() +
  scale_fill_viridis_c(option = "viridis", end = 0.8) +
  labs(title = "Distribution of Message Lengths", 
       x = "Message Length (characters)", 
       y = "Frequency") +
  scale_x_continuous(trans='log')
```

<img src="README_files/figure-gfm/length_distro-1.png"  />

``` r
# excerpt
#enron |>
#  mutate(content_length = nchar(content)) |>
#  summarize(max_length = max(content_length)) |> as.numeric()
```

### Truncate

``` r
require(tidyverse) # mutate
truncate_content <- function(content, max_length = 1000) {
  if (nchar(content) > max_length) {
    truncated <- substr(content, 1, max_length)
    last_space <- max(gregexpr(" ", truncated)[[1]])
    if (last_space > 0) {
      truncated <- substr(truncated, 1, last_space - 1)
    }
  } else {
    truncated <- content
  }
  return(truncated)
}
enron <- enron |> 
  mutate(content = sapply(content, truncate_content))
rm(truncate_content)
# excerpt
enron$content[23] |>
  str_wrap(width = 75) |>
  cat()
```

    ## review attachment final document provide starting review process

## Training/Testing Split

``` r
require(tidyverse)
set.seed(1)
train_ratio <- 0.8
enron <- enron %>%
  mutate(set = ifelse(runif(n()) < train_ratio, "train", "test"))
enron_train <- 
  enron %>% 
  filter(set == "train") %>% 
  select(-set)
enron_test  <- 
  enron %>% 
  filter(set == "test") %>% 
  select(-set)
#enron_train %>%
#  count(relevant) %>%
#  mutate(percentage = n / sum(n) * 100)
#enron_test %>%
#  count(relevant) %>%
#  mutate(percentage = n / sum(n) * 100)
rm(train_ratio
)
```

## Tokenize

``` r
require(tidyverse)
enron_train <- 
  enron_train |> 
  mutate(tokens = str_split(content, "\\s+")) # create vector of all words
vocab <- 
  enron_train$tokens |> 
  unlist() |> 
  table() |> 
  sort(decreasing = TRUE) |> 
  names()
word_to_index <- 
  setNames(seq_along(vocab), vocab)
enron_train <- enron_train |> 
  mutate(
    indices = map(tokens, ~ word_to_index[.x]),
    tokens = NULL)
na_indices <- 
  sapply(enron_train$indices, 
         function(x) any(is.na(x)))
enron_train <-
  enron_train[!na_indices, ]
enron_test <- 
  enron_test |> 
  mutate(tokens = str_split(content, "\\s+")) |> # tokenize
  mutate(indices = map(tokens, ~ word_to_index[.x]), tokens = NULL)
na_indices <- sapply(enron_test$indices, function(x) any(is.na(x)))
enron_test <- enron_test[!na_indices, ]
rm(na_indices)
rm(word_to_index)
# exerpt
enron_train |> filter(doc_id == "ENR.0001.0017.0141") |> pull(indices)
```

    ## [[1]]
    ##     review attachment      final   document    provide   starting     review 
    ##         50       1350        259        110        103       1117         50 
    ##    process 
    ##        117

## Convert to Tensors

``` r
require(torch)  # for tensor operations
set.seed(1)
torch_manual_seed(1)
#
# convert to tensors
#
pad <- length(vocab) + 1
indices_tensors_train <- 
  lapply(enron_train$indices, 
         torch_tensor, 
         dtype = torch_int64())
tensor_data_train <- 
  nn_utils_rnn_pad_sequence(indices_tensors_train, 
                            batch_first = TRUE,
                            padding_value = pad)
tensor_labels_train <- 
  torch_tensor(ifelse(enron$relevant == -1, 0, enron$relevant), 
               dtype = torch_int64())
indices_tensors_test <- 
  lapply(enron_test$indices, 
         torch_tensor, 
         dtype = torch_int64())
tensor_data_test <- 
  nn_utils_rnn_pad_sequence(indices_tensors_test, 
                            batch_first = TRUE,
                            padding_value = pad)
tensor_labels_test <- 
  torch_tensor(ifelse(enron_test$relevant == -1, 0, enron_test$relevant), 
               dtype = torch_int64())
rm(
  indices_tensors_train,
  indices_tensors_test,
  pad
)
#
# custom class and use it
#
custom_class <- dataset(
  initialize = function(tensors, labels) {
    self$tensors <- tensors  # Store the input tensors
    self$labels <- labels  # Store the labels
  },
  .getitem = function(index) {
    input_tensor <- self$tensors[index, ]
    label <- self$labels[index]
    return(list(input_tensor, label)) 
  },
  .length = function() {
    return(self$tensors$size(1))
  }
)
enron_tensor_dataset_train <- 
  custom_class(tensor_data_train, tensor_labels_train)
enron_tensor_dataset_test <- 
  custom_class(tensor_data_test, tensor_labels_test)
# exerpt
row_index <-
  which(enron_train$doc_id == "ENR.0001.0017.0141")
enron_tensor_dataset_train$.getitem(row_index)[[1]][enron_tensor_dataset_train$.getitem(row_index)[[1]] != length(vocab) + 1]
```

    ## torch_tensor
    ##    50
    ##  1350
    ##   259
    ##   110
    ##   103
    ##  1117
    ##    50
    ##   117
    ## [ CPULongType{8} ]

``` r
rm(row_index)
#enron_tensor_dataset_train$.getitem(row_index)[[2]]
```

### Declare Model

``` r
predict <- function(embedding_dim, 
                    n_hidden, 
                    batch_size,
                    learn_rate,
                    num_epochs) {
  require(torch)
  require(tidyverse)
  set.seed(1)
  torch_manual_seed(1)
  #
  # model variables
  #
  max_len <- tensor_data_train$size(2)
  #
  # model setup
  #
  model <- nn_module(
    initialize = function(embed_dim, n_hidden, max_len) {
      self$embedding <- nn_embedding(
        length(vocab) + 2, 
        embed_dim, 
        padding_idx = length(vocab) + 1)
      self$hidden <- nn_linear(embed_dim * max_len, n_hidden)
      self$dropout <- nn_dropout(p = 0.5)  # drop out with p probability
      self$output <- nn_linear(n_hidden, 1)
      self$sigmoid <- nn_sigmoid()
    },
    forward = function(x) {
      x <- self$embedding(x)  # embedding lookup
      x <- torch_flatten(x, start_dim = 2)  # flatten embeddings
      x <- self$hidden(x)  # Pass through the hidden layer
      x <- self$dropout(x)  # apply dropout
      x <- self$output(x)  # Pass through the output layer
      x <- self$sigmoid(x)  # Apply sigmoid to get probabilities
      return(x)
    }
  )
  #
  # instantiate
  #
  nn <- model(embedding_dim, n_hidden, max_len)
  rm(
    model, 
    embedding_dim,
    max_len, 
    n_hidden)
  #
  # dataloader
  #
  dataloader_train <- dataloader(
    dataset = enron_tensor_dataset_train,  # Your custom dataset
    batch_size = batch_size,         # The batch size you want to use
    shuffle = TRUE                   # Whether to shuffle the data
  )
  dataloader_test <- dataloader(
    dataset = enron_tensor_dataset_test,  # Your custom dataset
    batch_size = batch_size,         # The batch size you want to use
    shuffle = TRUE                   # Whether to shuffle the data
  )
  #
  # training
  #
  optimizer <- optim_adam(nn$parameters, lr = learn_rate)
  criterion <- nn_bce_loss()
  for (epoch in 1:num_epochs) {
    total_loss <- 0
    coro::loop(for (batch in dataloader_train) {
      batch_data <- 
        batch[[1]]$to(dtype = torch_int64()) + 
        torch_tensor(1, dtype = torch_int64())
      batch_labels <- batch[[2]]$to(dtype = torch_float32())
      optimizer$zero_grad()  # reset gradients
      output <- nn(batch_data) # forward pass
      loss <- criterion(output, batch_labels) # compute loss
      loss$backward() # backward pass and optimize
      optimizer$step()
      total_loss <- total_loss + loss$item()
    })
    cat("Epoch:", 
        epoch, 
        "Average Loss:", 
        total_loss / length(dataloader_train), 
        "\n")
  }
  nn$eval() # set model to evaluation mode
  with_no_grad({ # no need to compute gradients in eval mode
    train_scores <- 
      nn(tensor_data_train)$squeeze() # fwd pass, remove singleton dims
  })
  train_scores <- # create new variable
    as_array(train_scores) # convert scores tensor to r vector
  enron_train <- # update original dataset
    enron_train |> # pipe out original dataset
    mutate(score = train_scores) # add new column
  with_no_grad({ # no need to compute gradients in eval mode
    train_scores <- 
      nn(tensor_data_train)$squeeze() # fwd pass, remove singleton dims
  })
  train_scores <- # create new variable
    as_array(train_scores) # convert scores tensor to r vector
  enron_train <<- # update original dataset
    enron_train |> # pipe out original dataset
    mutate(score = train_scores) # add new column
  with_no_grad({
    test_scores <- 
      nn(tensor_data_test)$squeeze() # fwd pass, remove singleton dims
  })
  test_scores <- # create new variable
    as_array(test_scores) # convert scores tensor to r vector
  enron_test <<- # update original dataset
    enron_test |> # pipe out original dataset
    mutate(score = test_scores) # add new column
  rm(
    criterion, 
    optimizer, 
    num_epochs, 
    epoch,
    train_scores,
    test_scores)
  accuracy <- enron_test |>
    mutate(
      correct = (score > 0.5 & relevant == 1) | 
        (score <= 0.5 & relevant == 0)) |>
    summarize(correct_percentage = mean(correct) * 100) |>
    as.numeric()
  return(accuracy)
}
predict(embedding_dim = 64, # best 64
        n_hidden = 64, # best 64
        batch_size = 500, # best 500
        learn_rate = 0.001, # best 0.001
        num_epochs = 10) # best 10
```

    ## Epoch: 1 Average Loss: 3.879612 
    ## Epoch: 2 Average Loss: 26.06334 
    ## Epoch: 3 Average Loss: 26.76064 
    ## Epoch: 4 Average Loss: 20.05321 
    ## Epoch: 5 Average Loss: 14.95605 
    ## Epoch: 6 Average Loss: 10.79899 
    ## Epoch: 7 Average Loss: 7.020485 
    ## Epoch: 8 Average Loss: 2.215733 
    ## Epoch: 9 Average Loss: 1.732187 
    ## Epoch: 10 Average Loss: 1.197121

    ## [1] 55.9322

## Result Distribution

``` r
require(ggplot2) # for plotting
require(tidyverse) # for data manipulation
require(viridis)
ggplot(enron_test, aes(x = score, fill = ..count..)) +
  geom_histogram(binwidth = 0.1) +
  labs(title = "Histogram of Scores",
       x = "Score",
       y = "Frequency") +
  scale_fill_viridis_c(option = "viridis", end = 0.8)
```

<img src="README_files/figure-gfm/result_distribution-1.png"  />

### Result Sample

``` r
# exerpt
relevant_1 <- enron_test |>
  filter(relevant == 1) |>
  sample_n(5)
relevant_0 <- enron_test |>
  filter(relevant == 0) |>
  sample_n(5)
sampled_enron <- 
  bind_rows(relevant_1, relevant_0)
sampled_enron |>
  select(doc_id, relevant, score) |>
  as.data.frame()
```

    ##                doc_id relevant      score
    ## 1  ENR.0001.0045.0984        1 0.86490560
    ## 2  ENR.0001.0217.0701        1 0.40854099
    ## 3  ENR.0001.0037.0631        1 0.49969888
    ## 4  ENR.0001.0177.0856        1 0.86068857
    ## 5  ENR.0001.0118.0483        1 0.04761263
    ## 6  ENR.0001.0117.0087        0 0.47990373
    ## 7  ENR.0001.0113.0742        0 0.35656768
    ## 8  ENR.0001.0118.0421        0 0.29154062
    ## 9  ENR.0001.0117.0243        0 0.35441574
    ## 10 ENR.0001.0115.0166        0 0.44813871

``` r
rm(relevant_1,
   relevant_0,
   sampled_enron)
```
