"""
Note: you need to install the package from https://github.com/epfml/sent2vec and download twitter_unigrams.bin
model in order to be able to run this script. You also need NLTK pre-trained model to use a tokenizer.
"""
import sent2vec
import numpy as np
from nltk.tokenize import TweetTokenizer
import torch

from intent_detection.intent.mlp_classifier import MLPClassifier
from intent_detection.intent.utils import get_data_clinc150

torch.manual_seed(0)


tk = TweetTokenizer()
model = sent2vec.Sent2vecModel()
model.load_model('twitter_unigrams.bin')
uni_embs, vocab = model.get_unigram_embeddings()  # Return the full unigram embedding matrix

v = {}
for i in range(len(vocab)):
    v[vocab[i]] = uni_embs[i, :]


def sentence_embedding(sentence):
    emb = np.zeros(700)
    tokens = tk.tokenize(sentence)
    for t in tokens:
        if t.lower() in v:
            emb += v[t.lower()]
    emb /= len(tokens)
    return emb


def get_embeddings_data_s2v(data):
    emb = []
    for sample in data:
        emb.append(sentence_embedding(sample).tolist())
    return torch.FloatTensor(emb)


def get_embeddings_dataset_s2v(x_trn, x_tst, x_val):
    return get_embeddings_data_s2v(x_trn), get_embeddings_data_s2v(x_tst), get_embeddings_data_s2v(x_val)


def predict_sent2vec_supervised(model, query):
    emb = query.view(-1, query.shape[0]).requires_grad_()
    outputs = torch.nn.functional.softmax(model.forward(emb), 1)
    score, predicted = torch.max(outputs.data, 1)
    out = predicted[0].item()
    return out, score[0].item()


def evaluate_s2v_supervised(x_trn, y_trn, x_tst, y_tst, x_val, y_val=None, templates=None, dataset=None, emb_models=None, thr=None):
    emb_trn, emb_tst, emb_val = get_embeddings_dataset_s2v(x_trn, x_tst, x_val)
    correct, j = 0, 1
    model_nn = MLPClassifier()
    model_nn.fit(emb_trn, y_trn, x_val=emb_val, y_val=y_val, verbose=True, dataset_name=dataset)
    for sample, label in zip(emb_tst, y_tst):
        intent, _ = predict_sent2vec_supervised(model_nn, sample)
        correct += int(intent == label)
        percents = (j / len(x_tst)) * 100
        if np.isclose(percents % 1, 0):
            print(f"{percents} %")
        j += 1
    accuracy = correct / len(x_tst)
    return accuracy


if __name__ == '__main__':
    is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val, is_y_val, oos_x_trn, oos_y_trn, oos_x_tst, oos_y_tst, oos_x_val, oos_y_val = get_data_clinc150()

    is_acc = evaluate_s2v_supervised(is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val)

    x_train = is_x_trn + oos_x_trn
    y_train = np.concatenate([is_y_trn, oos_y_trn], axis=0)

    oos_acc = evaluate_s2v_supervised(x_train, y_train, oos_x_tst, oos_y_tst, oos_x_val)

    print(f"IS acc: {is_acc}, OOS acc: {oos_acc}")
