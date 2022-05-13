"""
Note: you need to install the package from https://github.com/epfml/sent2vec and download twitter_unigrams.bin
model in order to be able to run this script. You also need NLTK pre-trained model to use a tokenizer.
"""
from sklearn.model_selection import KFold
import sent2vec
import numpy as np
from nltk.tokenize import TweetTokenizer
import torch

from intent_detection.intent.mlp_classifier import MLPClassifier
from intent_detection.intent.utils import get_salon_data

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


def pair_emb(pair):
    x = sentence_embedding(pair[0])
    y = sentence_embedding(pair[1])
    return [x, y]


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
    samples, intents, labels = get_salon_data()

    k_folds = 4
    fold = 1
    x_train, y_train, x_test, y_test = {}, {}, {}, {}

    kfold = KFold(n_splits=k_folds, shuffle=True)
    for train_indices, test_indices in kfold.split(samples, labels):
        x_train[fold] = samples[train_indices].to_list()
        y_train[fold] = labels[train_indices]
        x_test[fold] = samples[test_indices].to_list()
        y_test[fold] = labels[test_indices]
        fold += 1

    total_s = 0
    for f in range(1, k_folds + 1):
        emb_trn = get_embeddings_data_s2v(x_train[f])
        emb_tst = get_embeddings_data_s2v(x_test[f])

        clf = MLPClassifier()
        clf.fit(emb_trn, y_train[f], batch_size=1, verbose=False)

        correct_s = 0
        for sample, emb, label in zip(x_test[f], emb_tst, y_test[f]):
            pred_s, _ = predict_sent2vec_supervised(clf, emb)
            correct_s += int(pred_s == label)

        acc_s = correct_s / len(x_test[f])
        total_s += acc_s

        print(f"Fold: {f}. Sent2vec accuracy: {acc_s:2.3f}")

    average_s = total_s / k_folds
    print(f"Average Sent2vec: {average_s:2.3f}")
