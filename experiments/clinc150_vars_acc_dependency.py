import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from intent_detection.intent.mlp_classifier import MLPClassifier
from intent_detection.intent.utils import (
    predict_cosine,
    predict_supervised,
    get_embeddings_dataset,
    get_data_clinc150,
    get_clinc150,
)
from intent_detection.intent.embedding import EmbeddingModel

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

trn_indices = {}
num_variations = len(
    np.where(is_y_trn == 0)[0]
)  # TODO: works only when there is equal number of variations
print(f"num variations: {num_variations}")

num_iter = 1
n_trn_indices = []
for k in range(num_iter):
    for i in range(1, num_variations + 1):
        trn_indices[i] = []
        for l in range(len(set(is_y_trn))):
            indices = np.where(is_y_trn == l)[0]
            sampled_indices = random.sample(indices.tolist(), i)

            for idx in sampled_indices:
                trn_indices[i].append(idx)
    n_trn_indices.append(trn_indices)

emb_models = ["all-MiniLM-L6-v2", "all-roberta-large-v1"]

for model in emb_models:

    print(f"Model: {model}")

    emb_model = EmbeddingModel(model)

    emb_trn, emb_tst, emb_val = get_embeddings_dataset(
        emb_model, is_x_trn, is_x_tst, is_x_val, "is"
    )

    xpoints, ypoints_s, ypoints_us = (
        list(range(1, num_variations + 1)),
        [0 for _ in range(1, num_variations + 1)],
        [0 for _ in range(1, num_variations + 1)],
    )
    total_time_us = [0 for _ in range(1, num_variations + 1)]
    total_time_s = [0 for _ in range(1, num_variations + 1)]

    for j in range(num_iter):
        for i in range(1, num_variations + 1):

            x_train = emb_trn[n_trn_indices[j][i], :]
            y_train = is_y_trn[n_trn_indices[j][i]]
            # x_test = samples["test"]
            x_test = emb_tst
            y_test = is_y_tst

            model_nn = MLPClassifier()
            model_nn.fit(x_train, y_train, verbose=False)

            correct_us, correct_s = 0, 0
            for sample, label in zip(x_test, y_test):
                start_time_em = time.time()
                # q = emb_model.predict(sample, convert_to_tensor=True)
                q = sample
                # print(f"embedding time: {(time.time() - start_time_em)}")

                start_time_us = time.time()
                _, pred_us, _ = predict_cosine(
                    emb_model, q, is_temp_trn, x_train, y_train
                )
                total_time_us[i - 1] += time.time() - start_time_us
                correct_us += int(pred_us == label)
                # print(f"unsupervised time: {(time.time() - start_time_us)}")

                start_time_s = time.time()
                pred_s, _ = predict_supervised(model_nn, q)
                total_time_s[i - 1] += time.time() - start_time_s
                correct_s += int(pred_s == label)
                # print(f"supervised time: {(time.time() - start_time_s)}")

            accuracy_s = 100 * (correct_s / len(x_test))
            accuracy_us = 100 * (correct_us / len(x_test))
            ypoints_s[i - 1] += accuracy_s
            ypoints_us[i - 1] += accuracy_us
            print(
                f"Number of variations {i}: Unsupervised accuracy: {accuracy_us:2.3f} %, Supervised accuracy: {accuracy_s:2.3f} %"
            )

    for i in range(num_variations):
        total_time_us[i] /= num_iter
        total_time_s[i] /= num_iter

    xpoints = np.array(xpoints)
    ypoints_s = np.array([y / num_iter for y in ypoints_s])
    ypoints_us = np.array([y / num_iter for y in ypoints_us])

    print("res_us", ",".join([str(a) for a in ypoints_us.tolist()]))
    print("res_s", ",".join([str(a) for a in ypoints_s.tolist()]))
    plt.plot(xpoints, ypoints_us, label=f"encoder: {model}, unsupervised")
    plt.plot(xpoints, ypoints_s, label=f"encoder: {model}, supervised")

    print("Inference time depending on number of variations\nUnsupervised")
    print(total_time_us)
    print("Supervised")
    print(total_time_s)

plt.xlabel("number of sentences [-]")
plt.ylabel("accuracy [%]")
plt.legend(loc="lower right")
plt.grid()

path = Path(__file__).parent / "plots"
if not path.exists():
    path.mkdir(parents=True)

plt.savefig(str(path / "num_vars_accuracy.png"), format="png")
plt.savefig(str(path / "num_vars_accuracy.eps"), format="eps")
plt.show()
