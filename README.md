# SMS Spam Classifier (GloVe + RNN)

An SMS spam detection model built with PyTorch and spaCy. It classifies text messages as **spam** or **ham** (not spam) using pre-trained GloVe word embeddings fed into a recurrent neural network.

## How it works

1. **Data**: the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset (~5,500 labeled SMS messages).
2. **Preprocessing**: messages are cleaned (punctuation/non-ASCII stripped) and tokenized with spaCy, removing stop words.
3. **Embeddings**: each token is mapped to a pre-trained 50-dimensional [GloVe](https://nlp.stanford.edu/projects/glove/) word vector (messages are padded/truncated to 20 tokens).
4. **Model**: a 2-layer RNN (`torch.nn.RNN`, hidden size 128) built in PyTorch, followed by a linear classification head.
5. **Training**: trained for 10 epochs with the Adam optimizer and cross-entropy loss, reaching ~96% training accuracy. The trained weights are saved to `rnn_model.pth`.

## Tech stack

- Python, Jupyter/Colab notebook
- PyTorch (model + training loop)
- spaCy (`en_core_web_sm`) for tokenization
- GloVe pre-trained word embeddings
- pandas / numpy / scikit-learn (`train_test_split`)

## Running it

The notebook (`Untitled13.ipynb`) was written for Google Colab and downloads its own data:

```bash
pip install torch spacy pandas numpy scikit-learn tqdm
python -m spacy download en_core_web_sm
```

Then run the notebook top to bottom — it will `wget` and unzip the SMS Spam Collection dataset and the GloVe 6B embeddings automatically before training.

## Note

The accuracy printed each epoch is training-set accuracy (the notebook splits off a test set with `train_test_split` but does not currently run a separate evaluation pass on it), so real-world generalization performance is untested. Adding a held-out evaluation step is a natural next improvement.
