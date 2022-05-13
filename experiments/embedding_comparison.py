import time
from scipy.stats import pearsonr

from intent_detection.intent.embedding import EmbeddingModel
from intent_detection.intent.utils import (
    get_acronyms,
    get_dataset_filenames,
    get_datasets,
    mae,
)

data = [
    ("-", "-", "STSb_test"),
    ("STSb_train", "STSb_dev", "STSb_test"),
    ("-", "-", "HP_test"),
    ("HP_train", "HP_dev", "HP_test"),
]
models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]

for train, devel, test in data:
    trn, dev, tst = get_acronyms(train, devel, test)

    train_files, dev_files, test_files = get_dataset_filenames(train, devel, test)
    x_trn, y_trn, x_dev, y_dev, x_tst, y_tst = get_datasets(train_files, dev_files, test_files)

    for model_name in models:
        model = EmbeddingModel(model_name)
        model.initialize()
        start_time = time.time()
        model.tune(x_trn, y_trn, x_dev, y_dev, epochs=10)
        trn_time = time.time() - start_time
        start_time = time.time()
        scores = model.pairwise_cosine_scores([x[0] for x in x_tst], [x[1] for x in x_tst])
        inf_time = time.time() - start_time
        pearson = pearsonr(y_tst, scores)[0]
        acc = 1 - mae(y_tst, scores)
        print(
            f"model: {model_name}, train: {train}, dev: {dev}, test: {test}, acc: {acc:2.3f}, pearson: {pearson:2.3f},"
            f"training time: {trn_time} s, inference time: {inf_time} s"
        )
