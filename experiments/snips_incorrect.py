import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from intent_detection.intent.utils import (
    cnf_matrix_bert,
    cnf_matrix_supervised,
    cnf_matrix_unsupervised,
    cnf_matrix_sent2vec,
    get_snips,
    get_data_snips,
)
from pathlib import Path


use_cuda = True
BERT_MODEL = "bert-base-uncased"
emb_models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]


def plot_confusion_matrices(m1, m2, m3):
    gs = gridspec.GridSpec(4, 4)

    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax0 = plt.subplot(gs[:2, :2])
    ax1 = plt.subplot(gs[:2, 2:])
    ax2 = plt.subplot(gs[2:4, 1:3])

    plot_confusion_matrix(
        ax0, cf=m1[emb_models[0]], title=f"unsupervised, {emb_models[0]}"
    )
    plot_confusion_matrix(
        ax1, cf=m2[emb_models[0]], title=f"supervised, {emb_models[0]}"
    )
    plot_confusion_matrix(ax2, cf=m3["BERT"], title=f"BERT-based")
    fig.tight_layout()

    path = Path(__file__).parent / "plots"
    if not path.exists():
        path.mkdir(parents=True)

    plt.savefig(str(path / "conf_matrix.eps"), format="eps")
    plt.show()


def plot_confusion_matrix(
    ax, cf, cbar=True, xyplotlabels=True, cmap="Blues", title=None
):
    sns.heatmap(
        cf,
        annot=True,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=True,
        yticklabels=True,
        square=True,
        ax=ax,
        annot_kws={"size": 9},
    )
    if xyplotlabels:
        ax.set_ylabel("True label", fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)


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

    dataset = "snips"
    (
        snips_x_trn,
        snips_y_trn,
        snips_x_tst,
        snips_y_tst,
        snips_x_val,
        snips_y_val,
    ) = get_snips()

    _, snips_labels_trn, _, snips_labels_tst, _, snips_labels_val = get_data_snips()

    matrices_us, matrices_s, matrices_b = {}, {}, {}
    categories = [
        "Play\nMusic",
        "AddTo\nPlay\nlist",
        "Rate\nBook",
        "Search\nScreening\nEvent",
        "Book\nRestau\nrant",
        "Get\nWeather",
        "Search\nCreative\nWork",
    ]
    if eval_unsup:
        hits, matrices_us = cnf_matrix_unsupervised(
            snips_x_trn,
            snips_y_trn,
            snips_x_tst,
            snips_y_tst,
            snips_x_val,
            snips_y_val,
            dataset=dataset,
            trn_intents=snips_labels_trn,
            true_intents=snips_labels_tst,
            emb_models=emb_models,
        )
        for hit in hits:
            print("Unsupervised", hit)

    if eval_sup:
        hits, matrices_s = cnf_matrix_supervised(
            snips_x_trn,
            snips_y_trn,
            snips_x_tst,
            snips_y_tst,
            snips_x_val,
            snips_y_val,
            dataset=dataset,
            trn_intents=snips_labels_trn,
            true_intents=snips_labels_tst,
            emb_models=emb_models,
        )
        for hit in hits:
            print("Supervised", hit)

    if eval_bert:
        hit, matrix = cnf_matrix_bert(
            snips_x_trn,
            snips_y_trn,
            snips_x_tst,
            snips_y_tst,
            snips_x_val,
            snips_y_val,
            dataset=dataset,
            trn_intents=snips_labels_trn,
            true_intents=snips_labels_tst,
            use_cuda=use_cuda,
        )
        print("BERT", hit)
        print(matrix)

    if eval_s2v:
        hit, matrix = cnf_matrix_sent2vec(
            snips_x_trn,
            snips_y_trn,
            snips_x_tst,
            snips_y_tst,
            snips_x_val,
            snips_y_val,
            dataset=dataset,
            trn_intents=snips_labels_trn,
            true_intents=snips_labels_tst,
        )
        print("sent2vec", hit)

    matrices_us[emb_models[0]] = np.array(
        [
            [74, 10, 0, 0, 1, 0, 1],
            [4, 120, 0, 0, 0, 0, 0],
            [0, 0, 79, 0, 1, 0, 0],
            [2, 1, 0, 92, 1, 0, 11],
            [0, 0, 0, 0, 91, 1, 0],
            [0, 0, 0, 1, 0, 103, 0],
            [8, 6, 5, 4, 0, 0, 84],
        ]
    )
    matrices_s[emb_models[0]] = np.array(
        [
            [79, 0, 0, 0, 0, 0, 7],
            [1, 123, 0, 0, 0, 0, 0],
            [0, 0, 80, 0, 0, 0, 0],
            [0, 0, 0, 94, 0, 0, 13],
            [0, 0, 0, 0, 92, 0, 0],
            [0, 0, 0, 0, 1, 103, 0],
            [5, 0, 0, 3, 0, 0, 99],
        ]
    )
    matrices_us[emb_models[1]] = np.array(
        [
            [78, 5, 0, 0, 0, 0, 3],
            [4, 120, 0, 0, 0, 0, 0],
            [0, 0, 80, 0, 0, 0, 0],
            [2, 0, 0, 97, 0, 0, 8],
            [0, 0, 0, 0, 92, 0, 0],
            [0, 0, 0, 0, 1, 103, 0],
            [9, 1, 1, 4, 0, 0, 92],
        ]
    )
    matrices_s[emb_models[1]] = np.array(
        [
            [81, 1, 0, 0, 0, 0, 4],
            [0, 123, 0, 0, 0, 0, 1],
            [0, 0, 80, 0, 0, 0, 0],
            [2, 0, 0, 99, 0, 0, 6],
            [0, 0, 0, 0, 92, 0, 0],
            [0, 0, 0, 0, 1, 103, 0],
            [7, 0, 0, 3, 1, 0, 96],
        ]
    )

    matrices_b["BERT"] = np.array(
        [
            [82, 1, 0, 0, 0, 0, 3],
            [0, 124, 0, 0, 0, 0, 0],
            [0, 0, 80, 0, 0, 0, 0],
            [0, 0, 0, 96, 0, 0, 11],
            [0, 0, 0, 0, 92, 0, 0],
            [0, 0, 0, 0, 1, 103, 0],
            [3, 0, 0, 1, 0, 0, 103],
        ]
    )
    if matrices_us and matrices_s:
        print(matrices_us[emb_models[0]])
        print(matrices_s[emb_models[0]])
        print(matrices_us[emb_models[1]])
        print(matrices_s[emb_models[1]])

        plot_confusion_matrices(matrices_us, matrices_s, matrices_b)
