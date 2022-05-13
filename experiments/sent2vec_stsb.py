"""
Note: you need to install the package from https://github.com/epfml/sent2vec and download twitter_unigrams.bin
model in order to be able to run this script. You also need NLTK pre-trained model to use a tokenizer.
"""
import sent2vec
import numpy as np
from nltk.tokenize import TweetTokenizer

from intent_detection.intent.utils import get_datasets, pearson

# train_file = 'stsbenchmark.tsv.gz'
# val_file = 'stsbenchmark.tsv.gz'
# test_file = 'stsbenchmark.tsv.gz'
train_file = "STS_train_2021_harry_potter.json"
val_file = "STS_dev_2021_harry_potter.json"
test_file = "STS_test_2021_harry_potter.json"

tk = TweetTokenizer()
model = sent2vec.Sent2vecModel()
model.load_model('twitter_unigrams.bin')
uni_embs, vocab = model.get_unigram_embeddings()  # Return the full unigram embedding matrix


v = {}
for i in range(len(vocab)):
    v[vocab[i]] = uni_embs[i, :]


def cosine_similarity(pairs):
    scores = []
    for pair in pairs:
        x, y = pair
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if np.isclose(norm_x * norm_y, 0):
            score = 0
        else:
            score = np.dot(x, y) / (norm_x * norm_y)
        scores.append(score)
    return np.array(scores)


def sentence_embedding(sentence):
    emb = np.zeros(700)
    tokens = tk.tokenize(sentence)
    for t in tokens:
        token = t.lower().strip()
        if token in v:
            emb += v[token]
        else:
            print(token)
    return emb / len(tokens)


def pair_emb(pair):
    x = sentence_embedding(pair[0])
    y = sentence_embedding(pair[1])
    return [x, y]


if __name__ == '__main__':
    x_trn, y_trn, x_dev, y_dev, x_tst, y_tst = get_datasets(train_file, val_file, test_file)

    for i in range(len(x_tst)):
        x_tst[i] = pair_emb(x_tst[i])

    pred = cosine_similarity(x_tst)
    mae = np.mean([abs(y_pred - y_true) for y_pred, y_true in zip(pred, y_tst)])
    acc = 1 - mae
    p = pearson(pred, y_tst)
    print(f"1-mae: {acc}. Pearson: {p}")
