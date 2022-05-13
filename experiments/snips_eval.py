import argparse
from intent_detection.intent.utils import (
    evaluate_bert,
    evaluate_unsupervised,
    evaluate_sent2vec,
    evaluate_supervised,
    get_snips,
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
    eval_bert, eval_sup, eval_unsup, eval_s2v, use_cuda = vars(parser.parse_args()).values()

    dataset = "snips"
    snips_x_trn, snips_y_trn, snips_x_tst, snips_y_tst, snips_x_val, snips_y_val = get_snips()

    if eval_bert:
        acc_bert = evaluate_bert(BERT_MODEL, snips_x_trn, snips_y_trn, snips_x_tst, snips_y_tst, snips_x_val,
                                 snips_y_val, dataset=dataset, use_cuda=use_cuda)
        print(f"BERT acc: {acc_bert}")
    if eval_unsup:
        acc_us, _ = evaluate_unsupervised(snips_x_trn, snips_y_trn, snips_x_tst, snips_y_tst, snips_x_val,
                                          snips_y_val, dataset=dataset, emb_models=emb_models)
        print(f"Unsupervised: {' '.join(['Model: {0}. Acc: {1}'.format(k, v) for k, v in acc_us.items()])}")
    if eval_sup:
        acc_s, _ = evaluate_supervised(snips_x_trn, snips_y_trn, snips_x_tst, snips_y_tst, snips_x_val,
                                       snips_y_val, dataset=dataset, emb_models=emb_models)
        print(f"Supervised: {' '.join(['Model: {0}. Acc: {1}'.format(k, v) for k, v in acc_s.items()])}")
    if eval_s2v:
        acc_s2v = evaluate_sent2vec(snips_x_trn, snips_y_trn, snips_x_tst, snips_y_tst, snips_x_val,
                                    snips_y_val, dataset=dataset)
        print(f"Sent2vec acc: {acc_s2v}")
