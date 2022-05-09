import torch
from pathlib import Path
import numpy as np
from intent_detection.intent.mlp_classifier import MLPClassifier
from intent_detection.intent.utils import (
    predict_cosine,
    predict_supervised,
    get_clinc150,
    get_data_clinc150,
)
from intent_detection.intent.embedding import EmbeddingModel
from sklearn.model_selection import KFold


emb_models = ["all-MiniLM-L6-v2"]


def reformat_train_test(x_train, y_train, x_test, y_test):
    return list(zip(x_train, y_train)), list(zip(x_test, y_test))


samples_trn, intents_trn, samples_tst, intents_tst, _, _ = get_clinc150("is")
_, labels_trn, _, labels_tst, _, _, _, _, _, _, _, _ = get_data_clinc150()

# samples_tst, intents_tst, labels_tst = [], [], []
samples = np.append(samples_trn, samples_tst, axis=0)
intents = np.append(intents_trn, intents_tst, axis=0)
labels = np.append(labels_trn, labels_tst, axis=0)

k_folds = 3
fold = 1
x_train, y_train, x_test, y_test = {}, {}, {}, {}

kfold = KFold(n_splits=k_folds, shuffle=True)
for train_indices, test_indices in kfold.split(samples, labels):
    x_train[fold] = samples[train_indices]
    y_train[fold] = labels[train_indices]
    x_test[fold] = samples[test_indices]
    y_test[fold] = labels[test_indices]
    fold += 1

for model in emb_models:
    emb_model = EmbeddingModel(model)
    print(f"Model: {model}")
    total_us, total_s = 0, 0

    for f in range(1, k_folds + 1):
        emb_trn = emb_model.predict(x_train[f], convert_to_tensor=True)
        emb_tst = emb_model.predict(x_test[f], convert_to_tensor=True)

        clf = MLPClassifier()
        clf.fit(emb_trn, y_train[f], verbose=False)

        correct_us, correct_s = 0, 0
        for sample, label in zip(emb_tst, y_test[f]):
            _, pred_us, _ = predict_cosine(
                emb_model, sample, x_train[f], emb_trn, y_train[f]
            )
            correct_us += int(pred_us == label)

            pred_s, _ = predict_supervised(clf, sample)
            correct_s += int(pred_s == label)

        acc_us = correct_us / len(x_test[f])
        acc_s = correct_s / len(x_test[f])

        total_us += acc_us
        total_s += acc_s
        print(
            f"Fold: {f}. Unsupervised accuracy: {acc_us:2.3f}, Supervised accuracy: {acc_s:2.3f}"
        )

    avr_us = total_us / k_folds
    avr_s = total_s / k_folds
    print(
        f"Embedding model {model}. Unsupervised average accuracy: {avr_us:2.3f}, Supervised average accuracy: {avr_s:2.3f}"
    )
