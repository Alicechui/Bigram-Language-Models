import argparse
import math
import numpy as np


BOS_MARKER = "<s>"
EOS_MARKER = "</s>"
UNKNOWN = "<UNK>"

# random number generator
rng = np.random.default_rng()


def preprocess_sentence(sent, vocab=None, markers=False):
    """
    Preprocess sent by tokenizing (splitting on whitespace),
    and converting tokens to lowercase. If vocab is provided, OOV words
    are replaced with UNKNOWN. If markers is True, sentence begin and
    end markers are inserted.

    :param sent: sentence to process, as a string
    :param vocab: if provided, oov words are replaced with UNKNOWN
    :param markers: if True, BOS and EOS markers are added
    :return: list of lowercase token strings, with optional OOV replacement
    and marker insertion
    """
    pass


def load_data(corpus_file, vocab=None, markers=False):
    """
    Read corpus file line-by-line and return a list of preprocessed sentences,
    where preprocessing includes tokenization and lowercasing tokens. If vocab
    is provided, OOV words are replaced with the UNKNOWN token. If markers is True,
    sentence begin and end markers are inserted.

    Notes:

    - vocab can only be provided when loading test data, after the vocabulary has
    been established

    - markers should be False when loading training data (the markers will be added later,
    in get_bigram_counts(), after the unigram counts have been calculated),
    and True when loading test data.

    :param corpus_file: file containing one sentence per line
    :param vocab: if provided, OOV words are replaced with UNKNOWN
    :param markers: if True, BOS and EOS markers are added
    :return: a list of lists representing preprocessed sentences
    """
    pass


def get_unigram_counts(training_data, remove_low_freq_words=False):
    """
    From the training data, get the vocabulary as a list of words,
    and the unigram counts represented as a dictionary with
    words as keys and frequencies as values. If remove_low_freq_words
    is True, any word with count == 1 is removed
    from the dictionary, and its count (1) is added to the count of
    the UNKNOWN token.

    :param training_data: list of lists of words, without sentence markers
    :param remove_low_freq_words: if True, transfer the count of words appearing
    only once to the UNKNOWN  token
    :return: a list of vocabulary words, and a dictionary of words:frequencies (optionally
    with low-frequency word counts transferred to the UNKNOWN token)
    """
    pass


def get_vocab_index_mappings(vocab):
    """
    Assign each word in the vocabulary an index for access into the bigram probability
    matrix. Create and return 2 mappings, one for the rows and one for the
    columns. The mappings are dictionaries with vocabulary words as keys, and
    indices as values. The indices should start at 0 and each word should have
    a unique index. Include a BOS index in the row mapping, and an EOS index
    in the column mapping.

    :param vocab: a list of vocabulary words
    :return: two dictionaries with index mappings
    """
    pass


def get_bigram_counts(training_data, row_idx, col_idx, vocab, laplace=False):
    """
    Create and return a 2D matrix containing the bigram counts in training_data,
    optionally using Laplace smoothing.

    Before updating the bigrams counts for a sentence:

    - replace words in the sentence that are not in the vocabulary with the UNKNOWN token
    - add BOS and EOS sentence markers

    Note: It is possible to have words in the training data that are not part of the
    vocabulary if remove_low_freq_words=True when calling get_unigram_counts().

    :param training_data: list of lists of words, without sentence markers
    :param row_idx: word:row_index mapping
    :param col_idx: word:col_index mapping
    :param vocab: list of vocabulary words
    :param laplace: if True, use Laplace smoothing
    :return: a 2D matrix containing the bigram counts, optionally with Laplace smoothing
    """
    pass


def counts_to_probs(bigram_counts):
    """
    Returns a 2D matrix containing the bigram probabilities.

    :param bigram_counts: 2D matrix of integer counts of bigrams
    :return: a 2D matrix containing the bigram probabilities
    """
    pass


def to_log10(bigram_probs):
    """
    Convert a probability matrix to a log 10 probability matrix.

    :param bigram_probs: probability matrix
    :return: log 10 probability matrix
    """
    pass


def generate_sentence(bigram_probs, row_idx, col_idx):
    """
    Generate a sentence with probabilities according to the distributions
    given by bigram_probs. The returned sentence is a list of words that
    starts with BOS and continues until EOS is generated.

    :param bigram_probs: bigram probability matrix (not log10 matrix)
    :param row_idx: index mapping for rows
    :param col_idx: index mapping for columns
    :return: a sentence as a list of words, generated using bigram_probs
    """
    pass


def get_sent_logprob(bigram_logprobs, row_idx, col_idx, sent):
    """
    Returns the log 10 probability of sent.

    :param bigram_logprobs: log10 bigram matrix
    :param row_idx: row index mapping
    :param col_idx: column index mapping
    :param sent: a preprocessed sentence with BOS and EOS markers
    :return: the log10 probability of sent
    """
    pass


def get_perplexity(bigram_logprobs, row_idx, col_idx, test_sentences):
    """
    Calculate the perplexity of the test sentences according to the given
    bigram model.

    Notes:

    - Keep in mind that log10 probabilities are given, to avoid underflow,
    which means that the formula needs to be adjusted.

    - Get the perplexity of each sentence individually

    - Be careful with calculating N - see J&M 3.2.1

    :param bigram_logprobs: bigram logprob matrix
    :param row_idx: row index mapping
    :param col_idx: column index mapping
    :param test_sentences: list of preprocessed test sentences with BOS and EOS markers
    :return: the perplexity of the test sentences according to the given bigram model
    """
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_file", help="corpus file")
    parser.add_argument("test_file", help="test file")
    return parser.parse_args()


def main(args):
    """
    Train a bigram model on a file given by args.corpus_file. Remove low-frequency
    words, and apply Laplace smoothing.

    Generate and print 3 sentences.

    Evaluate your model by calculating the perplexity of the test data
    in args.test_file. Print the perplexity value.

    :param args: command-line arguments (args.corpus_file, args.test_file)
    """
    pass


if __name__ == '__main__':
    main(parse_args())