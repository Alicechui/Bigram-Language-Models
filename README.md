# Bigram Language Model

This project implements a Bigram Language Model using Python. The purpose of the model is to predict the next word in a sentence based on the previous word using bigram probabilities.

## Project Structure

- `bigram_model.py`: This file contains the implementation of the Bigram Language Model. It includes methods for preprocessing sentences, loading data, calculating unigram and bigram counts, converting counts to probabilities, generating sentences, calculating sentence log probabilities, and calculating perplexity.

- `test_bigram_model.py`: This file contains test cases for validating the functionality and accuracy of the Bigram Language Model. It covers different aspects of the model, including data loading, probability calculations, sentence generation, and perplexity calculation.

## Usage

To use the Bigram Language Model in your own project, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/Alicechui/Bigram-Language-Models.git
   ```

2. Import the `bigram_model.py` module into your Python script:

   ```python
   from bigram_model import BigramLanguageModel
   ```

3. Create an instance of the `BigramLanguageModel` class:

   ```python
   model = BigramLanguageModel()
   ```

4. Load the data and train the model:

   ```python
   model.load_data("emma.txt")
   model.train()
   ```

5. Use the model to generate sentences:

   ```python
   sentence = model.generate_sentence(seed_word="The")
   print(sentence)
   ```

6. Calculate the perplexity score for the model:

   ```python
   perplexity = model.get_perplexity()
   print("Perplexity:", perplexity)
   ```

## Testing

To run the test cases, execute the `test_bigram_model.py` file:

```bash
python test_bigram_model.py
```

The tests cover various functionalities of the Bigram Language Model and ensure the correctness of the implementation.

## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

Please refer to the [documentation](README.md) for more information and usage examples.

