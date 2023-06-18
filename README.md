# Bigram-Language-Model

This project is focused on training a bigram language model using sentences extracted from Jane Austen's novel "Emma." The following methods are utilized in the project:

- `preprocess_sentence()`: This method preprocesses a given sentence by removing punctuation, converting it to lowercase, and splitting it into a list of words.

- `load_data()`: This method loads the text data from Jane Austen's novel "Emma" and returns a list of preprocessed sentences.

- `get_unigram_counts()`: This method calculates the counts of each unigram (individual word) in the dataset and returns a dictionary mapping each word to its count.

- `get_vocab_index_mappings()`: This method creates a vocabulary of unique words and assigns an index to each word for later use in the language model.

- `get_bigram_counts()`: This method calculates the counts of each bigram (pair of consecutive words) in the dataset and returns a dictionary mapping each bigram to its count.

- `counts_to_probs()`: This method converts the counts of bigrams to probabilities using Laplace smoothing and returns a dictionary mapping each bigram to its probability.

- `to_log10()`: This method converts the probabilities of bigrams to their log10 values for efficient computation.

- `generate_sentence()`: This method generates a new sentence using the trained Bigram model, starting from a given seed word.

- `get_sentence_logprob()`: This method calculates the log probability of a given sentence using the trained Bigram model.

- `get_perplexity()`: This method calculates the perplexity score, which measures how well the language model predicts the given dataset, based on the log probabilities of all the sentences in the dataset.

By implementing these methods, the project aims to build a bigram language model and evaluate its performance using perplexity.
