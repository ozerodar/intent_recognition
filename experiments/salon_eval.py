from intent_detection.intent.mlp_classifier import MLPClassifier
from intent_detection.intent.utils import (
    predict_cosine,
    predict_supervised,
    predict_sent2vec,
    get_salon_data,
)
from intent_detection.intent.embedding import EmbeddingModel
from sklearn.model_selection import KFold


emb_models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]


if __name__ == "__main__":
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

    for model in emb_models:
        total_sent2vec, total_us, total_s = 0, 0, 0
        emb_model = EmbeddingModel(model)
        print(f"Model: {model}")
        for f in range(1, k_folds + 1):
            emb_trn = emb_model.predict(x_train[f], convert_to_tensor=True)
            emb_tst = emb_model.predict(x_test[f], convert_to_tensor=True)

            clf = MLPClassifier()
            clf.fit(emb_trn, y_train[f], batch_size=1, verbose=False)

            correct_us, correct_s, correct_sent2vec = 0, 0, 0
            for sample, emb, label in zip(x_test[f], emb_tst, y_test[f]):
                _, pred_us, _ = predict_cosine(
                    emb_model, emb, x_train[f], emb_trn, y_train[f]
                )
                correct_us += int(pred_us == label)

                pred_s, _ = predict_supervised(clf, emb)
                correct_s += int(pred_s == label)

                # pred_sen2vec = predict_sent2vec(sample, x_train[f], y_train[f])
                # correct_sent2vec += int(pred_sen2vec == label)

            acc_us = correct_us / len(x_test[f])
            acc_sent2vec = correct_sent2vec / len(x_test[f])
            acc_s = correct_s / len(x_test[f])

            total_us += acc_us
            total_sent2vec += acc_sent2vec
            total_s += acc_s

            print(
                f"Fold: {f}. Sent2vec accuracy: {acc_sent2vec:2.3f}, Unsupervised accuracy: {acc_us:2.3f}, Supervised accuracy: {acc_s:2.3f} "
            )

        average_s2v = total_sent2vec / k_folds
        average_us = total_us / k_folds
        average_s = total_s / k_folds
        print(
            f"Average Sent2vec: {average_s2v:2.3f}, average unsupervised: {average_us:2.3f}, average supervised: {average_s:2.3f}"
        )
