import argparse
import numpy as np
from intent_detection.intent.utils import (
    evaluate_bert,
    evaluate_unsupervised,
    evaluate_sent2vec,
    evaluate_supervised,
    get_data_clinc150,
    get_clinc150,
)

use_cuda = True
BERT_MODEL = "bert-base-uncased"
# BERT_MODEL = 'roberta-large'
emb_models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-b", action="store_true", help="BERT", default=False)
    parser.add_argument("-s", action="store_true", help="Supervised", default=False)
    parser.add_argument("-us", action="store_true", help="Unsupervised", default=False)
    parser.add_argument("-s2v", action="store_true", help="Sent2vec", default=False)
    parser.add_argument("-cuda", action="store_true", help="Use cuda", default=False)
    parser.add_argument("-thr", action="store_true", help="Threshold-based evaluation", default=False)
    eval_bert, eval_sup, eval_unsup, eval_s2v, use_cuda, eval_thr = vars(parser.parse_args()).values()

    is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val, is_y_val, oos_x_trn, oos_y_trn, oos_x_tst, oos_y_tst, oos_x_val, oos_y_val = get_data_clinc150()
    is_temp_trn, is_labels_trn, is_temp_tst, is_labels_tst, is_temp_val, is_labels_val = get_clinc150("is")

    x_train = is_x_trn + oos_x_trn
    y_train = np.concatenate([is_y_trn, oos_y_trn], axis=0)

    acc_bert_is, acc_bert_oos, acc_us_is, acc_us_oos, acc_s_is, acc_s_oos = -1, -1, {}, {}, {}, {}

    if eval_bert:
        acc_bert_is = evaluate_bert(BERT_MODEL, is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val, is_y_val,
                                    dataset="is", use_cuda=use_cuda, epochs=30)
        acc_bert_oos = evaluate_bert(BERT_MODEL, x_train, y_train, oos_x_tst, oos_y_tst, oos_x_val, oos_y_val,
                                     dataset="oos", use_cuda=use_cuda, epochs=30)
        print(f"BERT: In-scope: {acc_bert_is}. Out-of-scope: {acc_bert_oos}.\n")

    if eval_unsup:
        acc_us_is, _ = evaluate_unsupervised(is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val, is_y_val, dataset="is",
                                             emb_models=emb_models)
        acc_us_oos, _ = evaluate_unsupervised(x_train, y_train, oos_x_tst, oos_y_tst, oos_x_val, oos_y_val,
                                              dataset="oos", emb_models=emb_models)
        print(
            f"Unsupervised: In-scope: {' '.join(['Model: {0}. Acc: {1}'.format(k, v) for k, v in acc_us_is.items()])}\n"
            f"Unsupervised: Out-of-scope: {' '.join(['Model: {0}. Acc: {1}'.format(k, v) for k, v in acc_us_oos.items()])}\n"
        )

    if eval_sup:
        acc_s_is, _ = evaluate_supervised(is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val, is_y_val, dataset="is",
                                          emb_models=emb_models)
        acc_s_oos, _ = evaluate_supervised(x_train, y_train, oos_x_tst, oos_y_tst, oos_x_val, oos_y_val, dataset="oos",
                                           emb_models=emb_models)
        print(f"Supervised: In-scope: {' '.join(['Model: {0}. Acc: {1}'.format(k, v) for k, v in acc_s_is.items()])}\n"
              f"Supervised: Out-of-scope: {' '.join(['Model: {0}. Acc: {1}'.format(k, v) for k, v in acc_s_oos.items()])}\n")
    if eval_s2v:
        acc_sent2vec_is = evaluate_sent2vec(is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val, is_y_val, dataset="is")
        acc_sent2vec_oos = evaluate_sent2vec(x_train, y_train, oos_x_tst, oos_y_tst, oos_x_val, oos_y_val,
                                             dataset="oos")
        print(f"Sent2vec: In-scope: {acc_sent2vec_is}. Out-of-scope: {acc_sent2vec_oos}")

    if eval_thr:
        THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        x_test = is_x_val + oos_x_val
        y_test = np.concatenate([is_y_val, oos_y_val], axis=0)

        for thr in THRESHOLDS:
            acc_us_oos_thr = evaluate_unsupervised(is_x_trn, is_y_trn, x_test, y_test, x_val=None,
                                                   y_val=None, dataset="is_thr", emb_models=emb_models,
                                                   thr=thr, templates=is_temp_trn)
            acc_s_is_thr = evaluate_supervised(is_x_trn, is_y_trn, x_test, y_test, x_val=None, y_val=None,
                                               dataset="is_thr", emb_models=emb_models, thr=thr)
            print(f"Threshold: {thr}. Unsupervised thr acc: {acc_us_oos_thr}. \nSupervised: {acc_s_is_thr}")
