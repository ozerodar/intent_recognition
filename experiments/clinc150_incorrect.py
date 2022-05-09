import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from intent_detection.intent.utils import (
    cnf_matrix_bert,
    cnf_matrix_supervised,
    cnf_matrix_unsupervised,
    cnf_matrix_sent2vec,
    get_clinc150,
    get_data_clinc150,
)
from pathlib import Path

use_cuda = True
emb_models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]


def plot_confusion_matrix(cf_matrix, labels, model_name, dataset_name):
    df_cm = pd.DataFrame(cf_matrix, index=labels, columns=labels)
    plt.figure(figsize=(40, 40))
    ax = sns.heatmap(df_cm, cmap="YlGnBu")
    # ax.tight_layout()
    path = Path(__file__).parent / "plots"
    plt.savefig(str(path / f"{dataset_name}_{model_name}_matrix.eps"), format="eps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-b", action="store_true", help="BERT", default=False)
    parser.add_argument("-s", action="store_true", help="Supervised", default=False)
    parser.add_argument("-us", action="store_true", help="Unsupervised", default=False)
    parser.add_argument("-s2v", action="store_true", help="Sent2vec", default=False)
    parser.add_argument("-cuda", action="store_true", help="Use cuda", default=False)
    eval_bert, eval_sup, eval_unsup, eval_s2v, use_cuda = vars(
        parser.parse_args()
    ).values()

    dataset = "is"
    (
        is_x_trn,
        is_y_trn,
        is_x_tst,
        is_y_tst,
        is_x_val,
        is_y_val,
        oos_x_trn,
        oos_y_trn,
        oos_x_tst,
        oos_y_tst,
        oos_x_val,
        oos_y_val,
    ) = get_data_clinc150()
    _, is_labels_trn, _, is_labels_tst, _, is_labels_val = get_clinc150("is")

    if eval_unsup:
        hits, matrices = cnf_matrix_unsupervised(
            is_x_trn,
            is_y_trn,
            is_x_tst,
            is_y_tst,
            is_x_val,
            is_y_val,
            dataset="is",
            trn_intents=is_labels_trn,
            true_intents=is_labels_tst,
            emb_models=emb_models,
        )
        for name, matrix in matrices.items():
            plot_confusion_matrix(
                matrix, set(is_labels_tst), model_name=name, dataset_name=dataset
            )
        for hit in hits:
            print("Unsupervised", hit)

    if eval_sup:
        hits, matrices = cnf_matrix_supervised(
            is_x_trn,
            is_y_trn,
            is_x_tst,
            is_y_tst,
            is_x_val,
            is_y_val,
            dataset="is",
            trn_intents=is_labels_trn,
            true_intents=is_labels_tst,
            emb_models=emb_models,
        )
        for name, matrix in matrices.items():
            plot_confusion_matrix(
                matrix, set(is_labels_tst), model_name=name, dataset_name=dataset
            )
        for hit in hits:
            print("Supervised", hit)

    if eval_bert:
        hit, matrix = cnf_matrix_bert(
            is_x_trn,
            is_y_trn,
            is_x_tst,
            is_y_tst,
            is_x_val,
            is_y_val,
            dataset="is",
            trn_intents=is_labels_trn,
            true_intents=is_labels_tst,
        )
        plot_confusion_matrix(
            matrix, set(is_labels_tst), model_name="bert", dataset_name=dataset
        )
        print("BERT", hit)

    if eval_s2v:
        hit, matrix = cnf_matrix_sent2vec(
            is_x_trn,
            is_y_trn,
            is_x_tst,
            is_y_tst,
            is_x_val,
            is_y_val,
            dataset="is",
            trn_intents=is_labels_trn,
            true_intents=is_labels_tst,
        )
        plot_confusion_matrix(
            matrix, set(is_labels_tst), model_name="sent2vec", dataset_name=dataset
        )
        print("sent2vec", hit)
    plt.show()
