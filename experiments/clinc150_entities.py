import numpy as np

from intent_detection.intent.mlp_classifier import MLPClassifier
from intent_detection.intent.utils import (
    predict_cosine,
    predict_supervised,
    get_data_clinc150,
    get_clinc150,
    get_embeddings_dataset,
)
from intent_detection.intent.embedding import EmbeddingModel

emb_models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]


def label_to_intent(lbl, lbls, intns):
    for i in range(len(lbls)):
        if lbls[i] == lbl:
            return intns[i]
    return None


if __name__ == "__main__":
    (
        is_x_trn,
        is_y_trn,
        is_x_tst,
        is_y_tst,
        is_x_val,
        is_y_val,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_data_clinc150()
    (
        is_temp_trn,
        is_labels_trn,
        is_temp_tst,
        is_labels_tst,
        is_temp_val,
        is_labels_val,
    ) = get_clinc150("is")

    for model in emb_models:
        emb_model = EmbeddingModel(model)
        emb_trn, _, _ = get_embeddings_dataset(
            emb_model, is_x_trn, is_x_tst, is_x_val, dataset="is"
        )

        print(f"Model: {model}")

        clf = MLPClassifier()
        clf.fit(emb_trn, is_y_trn, batch_size=64, verbose=False)

        is_temp_tst = [
            "is chocolate cake healthy to eat",
            "is broccoli healthy to eat",
            "i am terry",
            "i am natalie",
            "show me something funny about reptiles",
            "show me something funny about flamingos",
        ]
        is_labels_tst = [
            "nutrition_info",
            "nutrition_info",
            "change_user_name",
            "change_user_name",
            "tell_joke",
            "tell_joke",
        ]
        is_y_tst = [55, 55, 18, 18, 138, 138]

        x_test = emb_model.predict(is_temp_tst, convert_to_tensor=True)
        y_test = np.array(is_y_tst)

        correct_us, correct_s = 0, 0
        for sample, emb, label in zip(is_temp_tst, x_test, y_test):
            match, pred_us, _ = predict_cosine(
                emb_model, emb, is_temp_trn, emb_trn, is_y_trn
            )
            correct_us += int(pred_us == label)
            print(
                f"Unsupervised match: {match}, prediction: {label_to_intent(pred_us, is_y_trn, is_labels_trn)}, true: {label}"
            )

            pred_s, _ = predict_supervised(clf, emb)
            correct_s += int(pred_s == label)
            print(
                f"Supervised prediction: {label_to_intent(pred_s, is_y_trn, is_labels_trn)}, true: {label}"
            )

        acc_us = correct_us / len(x_test)
        acc_s = correct_s / len(x_test)

        print(
            f"Unsupervised accuracy: {acc_us:2.3f}, Supervised accuracy: {acc_s:2.3f}"
        )
