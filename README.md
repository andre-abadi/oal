“oal” - Occasional Active Learning
================
Andre Abadi
2024-08-21

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
  geom_histogram(binwidth = 2000,) +
  #scale_x_log10() +
  scale_fill_viridis_c(option = "viridis") +
  labs(title = "Distribution of Message Lengths", 
       x = "Message Length (characters)", 
       y = "Frequency")
```

<img src="README_files/figure-gfm/length_distro-1.png"  />

``` r
# excerpt
enron |>
  mutate(content_length = nchar(content)) |>
  summarize(max_length = max(content_length)) |> as.numeric()
```

    ## [1] 135148

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
enron_tensor_dataset_train$.getitem(row_index)[[1]][enron_tensor_dataset_train$.getitem(23)[[1]] != length(vocab) + 1]
```

    ## torch_tensor
    ##     50
    ##   1350
    ##    259
    ##    110
    ##    103
    ##   1117
    ##     50
    ##    117
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ##  11774
    ## ... [the output was truncated (use n=-1 to disable)]
    ## [ CPULongType{102} ]

``` r
#enron_tensor_dataset_train$.getitem(row_index)[[2]]
```

### Declare Model

``` r
require(torch)
require(tidyverse)
set.seed(1)
torch_manual_seed(1)
#
# model variables
#
embedding_dim <- 32  # Embedding dimension
max_len <- tensor_data_train$size(2)
n_hidden <- 32  # Number of hidden units
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
nn <- model(embedding_dim, n_hidden, max_len)
rm(
  model, 
  embedding_dim,
  max_len, 
  n_hidden)
#
# training variables
#
batch_size <- 500
optimizer <- optim_adam(nn$parameters, lr = 0.001)
num_epochs <- 20
#
# data loader
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
    total_loss <- total_loss + loss$item() # accumulate loss for monitoring
  })
  cat("Epoch:", 
      epoch, 
      "Average Loss:", 
      total_loss / length(dataloader_train), 
      "\n")
}
```

    ## Epoch: 1 Average Loss: 1.281696 
    ## Epoch: 2 Average Loss: 1.290685 
    ## Epoch: 3 Average Loss: 1.057882 
    ## Epoch: 4 Average Loss: 0.8255272 
    ## Epoch: 5 Average Loss: 0.6022261 
    ## Epoch: 6 Average Loss: 0.6175299 
    ## Epoch: 7 Average Loss: 0.6096246 
    ## Epoch: 8 Average Loss: 0.5145197 
    ## Epoch: 9 Average Loss: 0.4515073 
    ## Epoch: 10 Average Loss: 0.4429001 
    ## Epoch: 11 Average Loss: 0.3975128 
    ## Epoch: 12 Average Loss: 0.3718781 
    ## Epoch: 13 Average Loss: 0.3488004 
    ## Epoch: 14 Average Loss: 0.336427 
    ## Epoch: 15 Average Loss: 0.3272066 
    ## Epoch: 16 Average Loss: 0.2950471 
    ## Epoch: 17 Average Loss: 0.2958529 
    ## Epoch: 18 Average Loss: 0.2750439 
    ## Epoch: 19 Average Loss: 0.2609306 
    ## Epoch: 20 Average Loss: 0.2466744

``` r
nn$eval() # set model to evaluation mode
with_no_grad({
  train_scores <- 
    nn(tensor_data_train)$squeeze() # fwd pass, remove singleton dims
})
train_scores <- # create new variable
  as_array(train_scores) # convert scores tensor to r vector
enron_train <- # update original dataset
  enron_train |> # pipe out original dataset
  mutate(score = train_scores) # add new column
with_no_grad({
  train_scores <- 
    nn(tensor_data_train)$squeeze() # fwd pass, remove singleton dims
})
train_scores <- # create new variable
  as_array(train_scores) # convert scores tensor to r vector
enron_train <- # update original dataset
  enron_train |> # pipe out original dataset
  mutate(score = train_scores) # add new column
with_no_grad({
  test_scores <- 
    nn(tensor_data_test)$squeeze() # fwd pass, remove singleton dims
})
test_scores <- # create new variable
  as_array(test_scores) # convert scores tensor to r vector
enron_test <- # update original dataset
  enron_test |> # pipe out original dataset
  mutate(score = test_scores) # add new column
rm(
  criterion, 
  optimizer, 
  num_epochs, 
  epoch,
  train_scores,
  test_scores)
enron_test |>
  mutate(
    correct = (score > 0.5 & relevant == 1) | 
      (score <= 0.5 & relevant == 0)) |>
  summarize(correct_percentage = mean(correct) * 100) |>
  as.numeric()
```

    ## [1] 46.61017

## Result Distribution

``` r
require(ggplot2) # for plotting
require(tidyverse) # for data manipulation
require(viridis)
ggplot(enron_test, aes(x = score, fill = ..count..)) +
  geom_histogram(binwidth = 0.05) +
  labs(title = "Histogram of Scores",
       x = "Score",
       y = "Frequency") +
  scale_fill_viridis_c()
```

<img src="README_files/figure-gfm/result_distribution-1.png"  />

### Result Sample

``` r
# exerpt
relevant_1 <- enron_test |>
  filter(relevant == 1) |>
  sample_n(10)
relevant_0 <- enron_test |>
  filter(relevant == 0) |>
  sample_n(10)
sampled_enron <- 
  bind_rows(relevant_1, relevant_0)
sampled_enron |>
  select(doc_id, relevant, score) |>
  as.data.frame()
```

    ##                doc_id relevant     score
    ## 1  ENR.0001.0045.0984        1 0.2733521
    ## 2  ENR.0001.0217.0701        1 0.1813560
    ## 3  ENR.0001.0037.0631        1 0.3923280
    ## 4  ENR.0001.0177.0856        1 0.2400435
    ## 5  ENR.0001.0118.0483        1 0.1188278
    ## 6  ENR.0001.0116.0506        1 0.6233805
    ## 7  ENR.0001.0118.0304        1 0.2420841
    ## 8  ENR.0001.0177.0740        1 0.7098238
    ## 9  ENR.0001.0118.0408        1 0.1180888
    ## 10 ENR.0001.0194.0850        1 0.5128819
    ## 11 ENR.0001.0118.0216        0 0.4568646
    ## 12 ENR.0001.0113.0677        0 0.4061086
    ## 13 ENR.0001.0116.0747        0 0.7990165
    ## 14 ENR.0001.0158.0254        0 0.4958428
    ## 15 ENR.0001.0116.0638        0 0.7752228
    ## 16 ENR.0001.0117.0093        0 0.8625825
    ## 17 ENR.0001.0116.0616        0 0.6347539
    ## 18 ENR.0001.0116.0660        0 0.4728081
    ## 19 ENR.0001.0117.0054        0 0.8037809
    ## 20 ENR.0001.0116.0768        0 0.4720142

``` r
rm(relevant_1,
   relevant_0,
   sampled_enron)
```

## NN Function

Encapsulate the neural network training and prediction in a function
call for ease of tuning.

``` r
predict <- function(embedding_dim, 
                    n_hidden, 
                    batch_size,
                    learn_rate,
                    num_epochs) {
  return(50)
}
predict(embedding_dim = 64,
        n_hidden = 64,
        batch_size = 500,
        learn_rate = 0.001,
        num_epochs = 20)
```

    ## [1] 50
