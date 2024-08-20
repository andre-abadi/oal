“oal” - Occasional Active Learning
================
Andre Abadi
2024-08-20

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
  fread( # tidyverse
    "data/enron_edisco.csv", # file to read
    select = c("doc_id", "relevant", "content") # only these columns
  ) |> 
  as_tibble() |>
  filter(!is.na(relevant)) # drop rows where relevant = 0
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
enron |>
  mutate(content_length = nchar(content)) |>
  summarize(max_length = max(content_length)) |> as.numeric()
```

    ## [1] 135148

### Truncate

``` r
require(tidyverse) # mutate
truncate_content <- function(content, max_length = 80) {
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
enron$content[23] |>
  str_wrap(width = 75) |>
  cat()
```

    ## review attachment final document provide starting review process

## Tokenize

``` r
require(tidyverse)
enron <- 
  enron |> 
  mutate(tokens = str_split(content, "\\s+")) # create vector of all words
vocab <- 
  enron$tokens |> 
  unlist() |> 
  table() |> 
  sort(decreasing = TRUE) |> 
  names()
word_to_index <- 
  setNames(seq_along(vocab), vocab)
enron <- enron |> 
  mutate(
    indices = map(tokens, ~ word_to_index[.x]),
    tokens = NULL)
rm(word_to_index)
na_indices <- sapply(enron$indices, function(x) any(is.na(x)))
enron <-
  enron[!na_indices, ]
rm(na_indices)
enron$indices[23]
```

    ## [[1]]
    ##     review attachment      final   document    provide   starting     review 
    ##        177        634        358         37        150       1078        177 
    ##    process 
    ##        193

## Convert to Tensors

``` r
require(torch)  # for tensor operations
set.seed(1)
torch_manual_seed(1)
#
# convert to tensors
#
pad <- length(vocab) + 1
indices_tensors <- 
  lapply(enron$indices, 
         torch_tensor, 
         dtype = torch_int64())
tensor_data <- 
  nn_utils_rnn_pad_sequence(indices_tensors, 
                            batch_first = TRUE,
                            padding_value = pad)
tensor_labels <- 
  torch_tensor(ifelse(enron$relevant == -1, 0, enron$relevant), 
               dtype = torch_int64())
rm(
  indices_tensors,
  pad)
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
enron_tensor_dataset <- 
  custom_class(tensor_data, tensor_labels)
enron_tensor_dataset$.getitem(23)[[1]][enron_tensor_dataset$.getitem(23)[[1]] != length(vocab) + 1]
```

    ## torch_tensor
    ##   177
    ##   634
    ##   358
    ##    37
    ##   150
    ##  1078
    ##   177
    ##   193
    ## [ CPULongType{8} ]

``` r
enron_tensor_dataset$.getitem(23)[[2]]
```

    ## torch_tensor
    ## 0
    ## [ CPULongType{} ]

### Declare Model

``` r
require(torch)
set.seed(1)
torch_manual_seed(1)
#
# model variables
#
embedding_dim <- 64  # Embedding dimension
max_len <- tensor_data$size(2)
n_hidden <- 64  # Number of hidden units
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
    self$output <- nn_linear(n_hidden, 1)
    self$sigmoid <- nn_sigmoid()
  },
  forward = function(x) {
    x <- self$embedding(x)  # embedding lookup
    x <- torch_flatten(x, start_dim = 2)  # flatten embeddings
    x <- self$hidden(x)  # Pass through the hidden layer
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
batch_size <- 500  # Set your desired batch size
optimizer <- optim_adam(nn$parameters, lr = 0.001)
num_epochs <- 20
#
# training
#
dataloader <- dataloader(
  dataset = enron_tensor_dataset,  # Your custom dataset
  batch_size = batch_size,         # The batch size you want to use
  shuffle = TRUE                   # Whether to shuffle the data
)
criterion <- nn_bce_loss()
for (epoch in 1:num_epochs) {
  total_loss <- 0
  coro::loop(for (batch in dataloader) {
    batch_data <- 
      batch[[1]]$to(dtype = torch_int64()) + torch_tensor(1, dtype = torch_int64())
    batch_labels <- batch[[2]]$to(dtype = torch_float32())
    optimizer$zero_grad()  # Reset gradients
    output <- nn(batch_data) # Forward pass
    loss <- criterion(output, batch_labels) # Compute loss
    loss$backward() # Backward pass and optimize
    optimizer$step()
    total_loss <- total_loss + loss$item() # Accumulate loss for monitoring
  })
  cat("Epoch:", 
      epoch, 
      "Average Loss:", 
      total_loss / length(dataloader), 
      "\n")
}
```

    ## Epoch: 1 Average Loss: 0.7536084 
    ## Epoch: 2 Average Loss: 0.6971813 
    ## Epoch: 3 Average Loss: 0.6308283 
    ## Epoch: 4 Average Loss: 0.5601185 
    ## Epoch: 5 Average Loss: 0.5388337 
    ## Epoch: 6 Average Loss: 0.5363574 
    ## Epoch: 7 Average Loss: 0.4870708 
    ## Epoch: 8 Average Loss: 0.4772458 
    ## Epoch: 9 Average Loss: 0.4537937 
    ## Epoch: 10 Average Loss: 0.4220158 
    ## Epoch: 11 Average Loss: 0.4147417 
    ## Epoch: 12 Average Loss: 0.3889803 
    ## Epoch: 13 Average Loss: 0.3743793 
    ## Epoch: 14 Average Loss: 0.3453557 
    ## Epoch: 15 Average Loss: 0.3407117 
    ## Epoch: 16 Average Loss: 0.3172497 
    ## Epoch: 17 Average Loss: 0.2870797 
    ## Epoch: 18 Average Loss: 0.2746077 
    ## Epoch: 19 Average Loss: 0.2581357 
    ## Epoch: 20 Average Loss: 0.2439252

``` r
rm(
  criterion, 
  optimizer, 
  num_epochs, 
  epoch)
```

### Predict

``` r
require(torch)
require(tidyverse) # mutate
set.seed(1)
torch_manual_seed(1)
nn$eval() # set model to evaluation mode
with_no_grad({
  scores <- 
    nn(tensor_data)$squeeze() # fwd pass, remove singleton dims
})
scores <- # create new variable
  as_array(scores) # convert scores tensor to r vector
enron <- # update original dataset
  enron |> # pipe out original dataset
  mutate(score = scores) # add new column
#enron[c(5:15, 23), c("doc_id", "relevant", "score")] |> as.data.frame()
set.seed(1)
relevant_1 <- enron |>
  filter(relevant == 1) |>
  sample_n(10)
relevant_0 <- enron |>
  filter(relevant == 0) |>
  sample_n(10)
sampled_enron <- 
  bind_rows(relevant_1, relevant_0)
sampled_enron |>
  select(doc_id, relevant, score) |>
  as.data.frame()
```

    ##                doc_id relevant       score
    ## 1  ENR.0001.0192.0140        1 0.048807401
    ## 2  ENR.0001.0045.0296        1 0.705716908
    ## 3  ENR.0001.0117.0499        1 0.759014845
    ## 4  ENR.0001.0117.0202        1 0.949854195
    ## 5  ENR.0001.0115.0332        1 0.076412171
    ## 6  ENR.0001.0115.0219        1 0.301738471
    ## 7  ENR.0001.0113.0012        1 0.097056858
    ## 8  ENR.0001.0115.0373        1 0.007583975
    ## 9  ENR.0001.0118.0639        1 0.752520800
    ## 10 ENR.0001.0115.0244        1 0.286277562
    ## 11 ENR.0001.0116.0978        0 0.610589683
    ## 12 ENR.0001.0115.0978        0 0.150573462
    ## 13 ENR.0001.0155.0673        0 0.005512342
    ## 14 ENR.0001.0118.0158        0 0.200124443
    ## 15 ENR.0001.0120.0192        0 0.976750314
    ## 16 ENR.0001.0034.0874        0 0.030013280
    ## 17 ENR.0001.0113.0660        0 0.292265028
    ## 18 ENR.0001.0120.0701        0 0.672328949
    ## 19 ENR.0001.0116.0918        0 0.761235237
    ## 20 ENR.0001.0118.0488        0 0.615665257

``` r
rm(relevant_1,
   relevant_0,
   sampled_enron)
```

## Result Distribution

``` r
require(ggplot2) # for plotting
require(tidyverse) # for data manipulation
require(viridis)
ggplot(enron, aes(x = score, fill = ..count..)) +
  geom_histogram(binwidth = 0.01) +
  labs(title = "Histogram of Scores",
       x = "Score",
       y = "Frequency") +
  scale_fill_viridis_c()
```

<img src="README_files/figure-gfm/result_distribution-1.png"  />
