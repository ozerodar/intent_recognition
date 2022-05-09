import os
import csv
import gzip
import random
import numpy as np
import torch
import json
import requests
import pandas as pd
from typing import Union, List
from sentence_transformers import util
from sklearn.metrics import confusion_matrix

from intent_detection.intent.bert_classifier import BertClassifier
from intent_detection import DIR_DATA, DIR_CLIENTS, EXPERIMENTS_DIR_DATA, MLP_MODEL_NAME, settings
from intent_detection.intent.mlp_classifier import MLPClassifier
from intent_detection.intent.embedding import EmbeddingModel


DEFAULT_PROJECT_ID = settings.DEFAULT_PROJECT_ID
DEFAULT_TOKEN = settings.DEFAULT_TOKEN
IP = settings.IP
PORT = settings.PORT
OOS = 150

DATASETS = {
    "-": "-",
    "STSb_train": "stsbenchmark.tsv.gz",
    "STSb_dev": "stsbenchmark.tsv.gz",
    "STSb_test": "stsbenchmark.tsv.gz",
    "HP_train": "STS_train_2021_harry_potter.json",
    "HP_dev": "STS_dev_2021_harry_potter.json",
    "HP_test": "STS_test_2021_harry_potter.json",
}


def get_intent_idx(data, intent):
    if not data:
        return 0

    for sample in data:
        if sample["intent"] == intent:
            return sample["index"]
    return data[-1]["index"] + 1


def upload_intents(path, data):
    intents = []

    if path.exists():
        intents = json.load(path.open("rt"))

    for sentence, intent in data.items():
        label = get_intent_idx(intents, intent)
        intents.append({"text": sentence, "index": label, "intent": intent})

    with open(path, "w") as outfile:
        json.dump(intents, outfile, indent=2)


def get_acronym(split: Union[List[str], str]):
    return split if isinstance(split, str) else "+".join(data for data in split)


def get_acronyms(train, dev, test):
    return get_acronym(train), get_acronym(dev), get_acronym(test)


def get_data_filenames(data: Union[List[str], str]):
    if isinstance(data, list):
        return [get_data_filenames(subset) for subset in data]
    else:
        return DATASETS[data]


def get_dataset_filenames(train, dev, test):
    return get_data_filenames(train), get_data_filenames(dev), get_data_filenames(test)


def get_datasets(
    train: List[str] = None, dev: List[str] = None, test: List[str] = None
):
    _train = [train] if isinstance(train, str) else train
    _dev = [dev] if isinstance(dev, str) else dev
    _test = [test] if isinstance(test, str) else test

    _train, _dev, _test = _train or [], _dev or [], _test or []
    x_trn, y_trn, x_dev, y_dev, x_tst, y_tst = [], [], [], [], [], []
    for train_file in _train:
        x, y = get_data(train_file, "train")
        x_trn.extend(x)
        y_trn.extend(y)
    for dev_file in _dev:
        x, y = get_data(dev_file, "dev")
        x_dev.extend(x)
        y_dev.extend(y)
    for test_file in _test:
        x, y = get_data(test_file, "test")
        x_tst.extend(x)
        y_tst.extend(y)

    print(
        f"number of pairs - train: {len(y_trn)}, dev: {len(y_dev)}, test: {len(y_tst)}"
    )
    return x_trn, y_trn, x_dev, y_dev, x_tst, y_tst


def get_data(filename, split):
    path = EXPERIMENTS_DIR_DATA / filename
    if ".json" in filename:
        x, y = get_data_sts(path)
    else:
        if filename == "stsbenchmark.tsv.gz" and not os.path.exists(
            path
        ):  # TODO: add parameter of smth
            util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", str(path))
        x, y = get_data_csv(path, split)
    return x, y


def get_data_sts(path):
    x, y = [], []

    if os.path.exists(path):
        try:
            data = json.load(path.open("rt"))
            random.shuffle(data)
            for sample in data:
                sent1 = sample[0]
                sent2 = sample[1]
                score = sample[2]

                x.append([sent1, sent2])
                y.append(score)
        except json.decoder.JSONDecodeError:
            pass
    else:
        print("path doesn't exist {}".format(path))  # TODO: absolute path for DATA
    return x, y


def get_data_csv(path, split):
    x, y = [], []
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf8") as file:
            reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == split:
                    score = (
                        float(row["score"]) / 5.0
                    )  # Normalize score to range 0 ... 1
                    texts = [
                        row["sentence1"]
                        .replace("’", "'")
                        .replace("‚", "'")
                        .replace('"', "'")
                        .encode("utf-8")
                        .decode(),
                        row["sentence2"]
                        .replace("’", "'")
                        .replace("‚", "'")
                        .replace('"', "'")
                        .encode("utf-8")
                        .decode(),
                    ]
                    x.append(texts)
                    y.append(score)
    return x, y


def mae(y, y_hat):
    errors = [abs(y - yh) for y, yh in zip(y, y_hat)]
    return sum(errors) / len(errors)


def get_intents(client_id=None, split=None):
    if client_id is not None:
        file = DIR_DATA / "clients" / client_id / "intents.json"
    elif split is not None:
        file = DIR_DATA / f"IC_{split}_intents.json"
    else:
        return NotImplementedError

    data = json.load(file.open("rt"))
    templates = [element["text"] for element in data]
    labels_names = [element["intent"] for element in data]
    labels = np.array([element["index"] for element in data])
    return templates, labels_names, labels


def predict_cosine(emb_model, query, templates, embeddings, labels):
    cosine_scores = emb_model.cosine_scores(embeddings, query)
    match_idx = torch.argmax(cosine_scores).item()
    match = templates[match_idx] if templates else ""
    return match, labels[match_idx], torch.max(cosine_scores).item()


def predict_supervised(model, query):
    emb = query.view(
        -1, query.shape[0]
    ).requires_grad_()  # TODO: move to model_nn.predict()
    outputs = torch.nn.functional.softmax(model.forward(emb), 1)
    score, predicted = torch.max(outputs.data, 1)
    out = predicted[0].item()
    return out, score[0].item()


def load_embeddings(client_id, model):  # only needed for an unsupervised approach
    path = DIR_CLIENTS / client_id / f"{model}_embeddings.pt"
    if path.exists():
        return torch.load(path)


def get_split_clinc150(split, part):
    file = EXPERIMENTS_DIR_DATA / "clinc150" / f"{part}_{split}.json"
    data = json.load(file.open("rt"))
    x, y, y_names = [], [], []
    for el in data:
        x.append(el[0])
        y.append(el[1])
    y = np.array(y)
    return x, y


def get_clinc150(part):
    x_trn, y_trn = get_split_clinc150("train", part)
    x_tst, y_tst = get_split_clinc150("test", part)
    x_val, y_val = get_split_clinc150("val", part)
    return x_trn, y_trn, x_tst, y_tst, x_val, y_val


def get_data_clinc150():
    is_x_trn, is_y_trn, is_x_tst, is_y_tst, is_x_val, is_y_val = get_clinc150("is")
    is_y_trn, is_y_tst, is_y_val = get_labels(is_y_trn, is_y_tst, is_y_val)
    oos_x_trn, oos_y_trn, oos_x_tst, oos_y_tst, oos_x_val, oos_y_val = get_clinc150(
        "oos"
    )
    oos_y_trn, oos_y_tst, oos_y_val = get_labels(
        oos_y_trn, oos_y_tst, oos_y_val, is_y_trn[-1] + 1
    )
    return (
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
    )


def get_embeddings(emb_model, data, split, dataset):
    emb_path = (
        EXPERIMENTS_DIR_DATA / f"{dataset}_embeddings_{split}_{emb_model.model_name}.pt"
    )
    if emb_path.exists():
        emb = torch.load(emb_path)
    else:
        emb = emb_model.predict(data, convert_to_tensor=True)
        torch.save(emb, emb_path)
    return emb


def get_embeddings_dataset(model, x_trn, x_tst, x_val, dataset):
    emb_trn = get_embeddings(model, x_trn, "train", dataset)
    emb_tst = get_embeddings(model, x_tst, "test", dataset)
    if x_val is not None:
        emb_val = get_embeddings(model, x_val, "val", dataset)
    else:
        emb_val = None
    return emb_trn, emb_tst, emb_val


def get_labels(y_trn, y_tst, y_val, label=None):
    if label is None:
        i = 0
        d = {}
        for name in y_trn:
            if name not in d:
                d[name] = i
                i += 1
    else:
        d = {name: label for name in set(y_trn)}
    y_trn = np.array([d[l] for l in y_trn])
    y_tst = np.array([d[l] for l in y_tst])
    y_val = np.array([d[l] for l in y_val])
    return y_trn, y_tst, y_val


def get_split_snips(split):
    file_x = EXPERIMENTS_DIR_DATA / "snips" / split / "seq.in"
    file_y = EXPERIMENTS_DIR_DATA / "snips" / split / "label"

    x, y = [], []
    with file_x.open("rt") as f:
        for line in f:
            x.append(line[:-1])

    with file_y.open("rt") as f:
        for line in f:
            y.append(line[:-1])

    return x, y


def get_data_snips():
    x_trn, y_trn = get_split_snips("train")
    x_tst, y_tst = get_split_snips("test")
    x_val, y_val = get_split_snips("val")
    return x_trn, y_trn, x_tst, y_tst, x_val, y_val


def get_snips():
    x_trn, y_trn, x_tst, y_tst, x_val, y_val = get_data_snips()
    y_trn, y_tst, y_val = get_labels(y_trn, y_tst, y_val)
    return x_trn, y_trn, x_tst, y_tst, x_val, y_val


def evaluate_unsupervised(
    x_trn,
    y_trn,
    x_tst,
    y_tst,
    x_val,
    y_val=None,
    templates=None,
    dataset=None,
    emb_models=None,
    thr=None,
):
    accuracies, accuracies_oos = {}, {}
    for model_name in emb_models:
        emb_model = EmbeddingModel(model_name)
        emb_trn, emb_tst, emb_val = get_embeddings_dataset(
            emb_model, x_trn, x_tst, x_val, dataset
        )
        correct, correct_is, correct_oos, ctr_is, ctr_oos = 0, 0, 0, 0, 0
        for sample, s, label in zip(emb_tst, x_tst, y_tst):
            match, pred, score = predict_cosine(
                emb_model, sample, templates, emb_trn, y_trn
            )
            # print(s, match, score)
            if thr is not None and score < thr:
                pred = OOS
            if label == OOS:
                correct_oos += int(pred == label)
                ctr_oos += 1
            else:
                correct_is += int(pred == label)
                ctr_is += 1
            correct += int(pred == label)
        if thr is not None:
            accuracies[model_name] = correct_is / ctr_is
            accuracies_oos[model_name] = correct_oos / ctr_oos
        else:
            accuracies[model_name] = correct / len(x_tst)
    return accuracies, accuracies_oos


def evaluate_supervised(
    x_trn,
    y_trn,
    x_tst,
    y_tst,
    x_val=None,
    y_val=None,
    templates=None,
    dataset=None,
    emb_models=None,
    thr=None,
):
    if emb_models is None:
        emb_models = []
    accuracies, accuracies_oos = {}, {}
    for model_name in emb_models:
        emb_model = EmbeddingModel(model_name)
        emb_trn, emb_tst, emb_val = get_embeddings_dataset(
            emb_model, x_trn, x_tst, x_val, dataset
        )
        model_nn = MLPClassifier()
        model_nn.fit(
            emb_trn,
            y_trn,
            x_val=emb_val,
            y_val=y_val,
            verbose=False,
            dataset_name=dataset,
        )
        correct, correct_is, correct_oos, ctr_oos, ctr_is = 0, 0, 0, 0, 0
        for sample, label in zip(emb_tst, y_tst):
            pred, score = predict_supervised(model_nn, sample)
            if thr is not None and score < thr:
                pred = OOS
            if label == OOS:
                correct_oos += int(pred == label)
                ctr_oos += 1
            else:
                correct_is += int(pred == label)
                ctr_is += 1
            correct += int(pred == label)
        if thr is not None:
            accuracies[model_name] = correct_is / ctr_is
            accuracies_oos[model_name] = correct_oos / ctr_oos
        else:
            accuracies[model_name] = correct / len(x_tst)
    return accuracies, accuracies_oos


def evaluate_bert(
    model_name,
    x_trn,
    y_trn,
    x_tst,
    y_tst,
    x_val,
    y_val,
    templates=None,
    dataset=None,
    use_cuda=False,
    **kwargs,
):
    model_nn = BertClassifier(
        output_dim=len(set(y_trn)), model_name=model_name, use_cuda=use_cuda
    )
    if use_cuda and torch.cuda.is_available():
        model_nn = model_nn.cuda()

    path = f"{model_nn.path}_{dataset}"
    if os.path.exists(path):
        model_nn.load_state_dict(torch.load(path))
        model_nn.eval()
    else:
        model_nn.fit(
            x_trn,
            y_trn,
            x_val=x_val,
            y_val=y_val,
            verbose=True,
            dataset_name=dataset,
            **kwargs,
        )
    return model_nn.evaluate(x_tst, y_tst)


def similarity_sent2vec(sentences1, sentences2):
    data1 = '["{}"]'.format('" ,"'.join(sentences1))
    data2 = '["{}"]'.format('" ,"'.join(sentences2))

    data = f'"sentences1": {data1}, "sentences2": {data2}'
    data = "{" + data + "}"
    response = requests.post(
        url="http://{}:{}/api/qa/similarity/pairwise_sentences?project_id={}".format(
            IP, PORT, DEFAULT_PROJECT_ID
        ),
        data=data.encode("utf-8"),
        headers={"Authorization": DEFAULT_TOKEN},
    )
    return response.json()
    # return max(response.json(), key=lambda x:x['confidence'])


def predict_sent2vec(sentences1, sentences2, labels):
    res = similarity_sent2vec(sentences1, sentences2)
    return [labels[sentences2.index(r["sentence"])] for r in res]


def get_salon_data():
    csv_file = str(EXPERIMENTS_DIR_DATA / "HSBQuestions.csv")
    df = pd.read_csv(csv_file)
    samples = df["Rephrased Question"]  # you can also use df['column_name']
    intents = df["Intent"]
    subinents = df["Subintent"]

    intents = [
        subintent if subintent != "None" else intent
        for intent, subintent in zip(intents, subinents)
    ]
    d = {intent: num for num, intent in enumerate(set(intents))}
    labels = np.array([d[intent] for intent in intents])
    return samples, intents, labels


def evaluate_sent2vec(
    x_trn, y_trn, x_tst, y_tst, x_val, y_val, templates=None, dataset=None
):
    correct = 0
    pred_sen2vec = predict_sent2vec(x_tst, x_trn, y_trn)
    for pred, l in zip(pred_sen2vec, y_tst):
        correct += int(pred == l)
    return correct / len(x_tst)


def cnf_matrix_unsupervised(
    x_trn,
    y_trn,
    x_tst,
    y_tst,
    x_val,
    y_val,
    trn_intents,
    true_intents,
    dataset=None,
    emb_models=None,
):
    hits, matrices = [], {}
    for model_name in emb_models:
        hit = {}

        emb_model = EmbeddingModel(model_name)
        emb_trn, emb_tst, emb_val = get_embeddings_dataset(
            emb_model, x_trn, x_tst, x_val, dataset
        )
        prediction = []
        for sample, s, label, l in zip(emb_tst, x_tst, y_tst, true_intents):
            match, pred, _ = predict_cosine(emb_model, sample, x_trn, emb_trn, y_trn)

            idx = np.where(y_trn == pred)[0][0]
            result = trn_intents[idx]
            prediction.append(pred)

            if pred == label:
                if result in hit:
                    hit[result] += 1
                else:
                    hit[result] = 1
            else:
                print(f"miss. sentence: {s}, match: {match}. True: {l}, Pred: {result}")
        conf_matrix = confusion_matrix(
            y_true=np.array(y_tst), y_pred=np.array(prediction)
        )
        matrices[model_name] = conf_matrix
        hits.append(dict(sorted(hit.items(), key=lambda item: item[1], reverse=False)))
    return hits, matrices


def cnf_matrix_supervised(
    x_trn,
    y_trn,
    x_tst,
    y_tst,
    x_val,
    y_val,
    trn_intents,
    true_intents,
    templates=None,
    dataset=None,
    emb_models=None,
):
    hits, matrices = [], {}
    for model_name in emb_models:
        hit = {}
        emb_model = EmbeddingModel(model_name)
        emb_trn, emb_tst, emb_val = get_embeddings_dataset(
            emb_model, x_trn, x_tst, x_val, dataset
        )
        model_nn = MLPClassifier()
        model_nn.fit(
            emb_trn,
            y_trn,
            x_val=emb_val,
            y_val=y_val,
            verbose=True,
            dataset_name=dataset,
        )
        prediction = []
        for sample, label in zip(emb_tst, y_tst):
            pred, _ = predict_supervised(model_nn, sample)
            idx = np.where(y_trn == pred)[0][0]
            result = trn_intents[idx]
            prediction.append(pred)

            if pred == label:
                if result in hit:
                    hit[result] += 1
                else:
                    hit[result] = 1
        conf_matrix = confusion_matrix(y_true=y_tst, y_pred=np.array(prediction))
        matrices[model_name] = conf_matrix
        hits.append(dict(sorted(hit.items(), key=lambda item: item[1], reverse=False)))
    return hits, matrices


def cnf_matrix_bert(
    x_trn,
    y_trn,
    x_tst,
    y_tst,
    x_val,
    y_val,
    trn_intents,
    true_intents,
    model_name=None,
    templates=None,
    dataset=None,
    use_cuda=False,
):
    model_nn = BertClassifier(
        output_dim=len(set(y_trn)), model_name=model_name, use_cuda=use_cuda
    )
    if use_cuda and torch.cuda.is_available():
        model_nn = model_nn.cuda()

    path = f"{model_nn.path}_{dataset}"
    if os.path.exists(path):
        model_nn.load_state_dict(torch.load(path))
        model_nn.eval()
    else:
        model_nn.fit(
            x_trn, y_trn, x_val=x_val, y_val=y_val, verbose=True, dataset_name=dataset
        )

    hit = {}
    prediction = []
    output = model_nn.predict(x_tst, y_tst)
    for pred, label in zip(output, y_tst):
        idx = np.where(y_trn == pred)[0][0]
        result = trn_intents[idx]
        prediction.append(pred)

        if pred == label:
            if result in hit:
                hit[result] += 1
            else:
                hit[result] = 1
    conf_matrix = confusion_matrix(y_true=np.array(y_tst), y_pred=np.array(prediction))
    return (
        dict(sorted(hit.items(), key=lambda item: item[1], reverse=False)),
        conf_matrix,
    )


def cnf_matrix_sent2vec(
    x_trn,
    y_trn,
    x_tst,
    y_tst,
    x_val,
    y_val,
    trn_intents,
    true_intents,
    model_name=None,
    templates=None,
    dataset=None,
):
    prediction = []
    hit = {}
    pred_sen2vec = predict_sent2vec(x_tst, x_trn, y_trn)
    for p, label in zip(pred_sen2vec, y_tst):
        idx = np.where(y_trn == p)[0][0]
        result = trn_intents[idx]
        prediction.append(p)
        if p == label:
            if result in hit:
                hit[result] += 1
            else:
                hit[result] = 1
    hit = dict(sorted(hit.items(), key=lambda item: item[1], reverse=False))
    conf_matrix = confusion_matrix(y_true=np.array(y_tst), y_pred=np.array(prediction))
    return hit, conf_matrix
