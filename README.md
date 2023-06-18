# Bigram-Language-Model

The project aims to train a bigram language model using sentences extracted from Jane Austen's novel "Emma." The following methods are employed in the project:

1. preprocess_sentence(): This method preprocesses a given sentence by removing punctuation, converting it to lowercase, and splitting it into a list of words.

2. load_data(): This method loads the text data from Jane Austen's novel "Emma" and returns a list of preprocessed sentences.

3. get_unigram_counts(): This method calculates the counts of each unigram (individual word) in the dataset and returns a dictionary mapping each word to its count.

4. get_vocab_index_mappings(): This method creates a vocabulary of unique words and assigns an index to each word for later use in the language model.

5. get_bigram_counts(): This method calculates the counts of each bigram (pair of consecutive words) in the dataset and returns a dictionary mapping each bigram to its count.

6. counts_to_probs(): This method converts the counts of bigrams to probabilities using Laplace smoothing and returns a dictionary mapping each bigram to its probability.

7. to_log10(): This method converts the probabilities of bigrams to their log10 values for efficient computation.

8. generate_sentence(): This method generates a new sentence using the trained bigram model, starting from a given seed word.

9. get_sentence_logprob(): This method calculates the log-probability of a given sentence using the trained bigram model.

10. get_perplexity(): This method calculates the perplexity score, a measure of how well the language model predicts the given dataset, based on the log-probabilities of all the sentences in the dataset.

By implementing these methods, the project aims to build a bigram language model, evaluate its performance using perplexity.
