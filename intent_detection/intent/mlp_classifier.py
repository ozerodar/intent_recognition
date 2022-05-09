"""
    intent_detection.mlp_classifier
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Intent Classifier class that classifier an utterance with a label
    @author: Daria Ozerova
"""
import os
import json
import subprocess
import tempfile
import requests

import torch
from torch import Tensor
import torch.nn as nn

from intent_detection import (
    DIR_HANDLERS,
    DIR_DATA,
    MLP_MODEL_NAME,
    DIR_MODELS,
    MODEL_STORE,
    settings,
)
from intent_detection.services.s3 import upload_file

torch.manual_seed(0)


def create_dataloader(x, y, batch_size=16):
    return torch.utils.data.DataLoader(
        dataset=list(zip(x, y)), batch_size=batch_size, shuffle=True
    )


class MLPClassifier(nn.Module):
    def __init__(self):
        self.input_dim = None
        self.output_dim = None
        self.linear1 = None
        self.linear2 = None
        self.linear = None
        self.dropout = None
        self.model_name = MLP_MODEL_NAME
        self.path = DIR_MODELS / f"{MLP_MODEL_NAME}.pt"
        self.initialized = False

    def initialize(self, input_dim, output_dim):
        if not self.initialized:
            super().__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.dropout = nn.Dropout(0.25)
            self.linear1 = nn.Linear(self.input_dim, 200)
            self.linear2 = nn.Linear(200, self.output_dim)
            self.initialized = True

    def forward(self, x0):
        dropout_output = self.dropout(x0)
        x1 = torch.relu(self.linear1(dropout_output))
        x2 = self.linear2(x1)
        return x2

    def get_accuracy(self, test_loader):
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = self.forward(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum()

        return 100 * correct / total

    def fit(self, x_train, y_train, **kwargs):
        batch_size = kwargs.get("batch_size", 64) if len(x_train) > 100 else 1
        n_iter = kwargs.get("n_iter", 5000)
        learning_rate = kwargs.get("l_rate", 0.001)
        weight_decay = kwargs.get("weight_decay", 0.000001)
        verbose = kwargs.get("verbose", False)
        num_epochs = kwargs.get("epoch") or int(n_iter / (len(x_train) / batch_size))
        output_path = kwargs.get("path") or self.path
        x_val = kwargs.get("x_val")
        y_val = kwargs.get("y_val")

        prev_acc = -1

        self.initialize(x_train.shape[1], len(set(y_train)))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=self.parameters(), weight_decay=weight_decay, lr=learning_rate
        )
        train_loader = create_dataloader(
            x_train, y_train, batch_size
        )  # TODO: move to notebooks and do in intent
        val_loader = None
        if isinstance(x_val, Tensor):
            val_loader = create_dataloader(x_val, y_val, batch_size)

        iteration = 0
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):

                optimizer.zero_grad()  # Clear gradients w.r.t. parameters
                outputs = self.forward(images)
                loss = criterion(
                    outputs, labels
                )  # Calculate Loss: softmax --> cross entropy loss
                loss.backward()  # Getting gradients w.r.t. parameters
                optimizer.step()  # Updating parameters

                if iteration % 500 == 0 and verbose:
                    val_acc = -1
                    if val_loader:
                        val_acc = self.get_accuracy(val_loader)
                        if val_acc > prev_acc:
                            # break
                            # model_scripted = torch.jit.script(self)  # Export to TorchScript
                            # model_scripted.save(output_path)  # Save model
                            prev_acc = val_acc
                    print(
                        f"Iteration: {iteration}. Loss: {loss.item()}. Val acc: {val_acc}."
                    )
                iteration += 1

        # model_scripted = torch.jit.script(self)  # Export to TorchScript
        # model_scripted.save(output_path)  # Save model


class MlpClassifierWrapper(MLPClassifier):
    def __init__(self, server):
        super().__init__()
        self.model_name = MLP_MODEL_NAME
        self.url = f"{settings.IP_EC2}:{settings.PORT_INF}"
        self.server = server

    def register_model(self, client_id, embeddings, intents, labels, **kwargs):
        output_path = DIR_DATA / "clients" / client_id / f"{MLP_MODEL_NAME}.pt"

        model = MLPClassifier()
        model.fit(embeddings, labels, path=output_path, **kwargs)
        self.archive_model(
            intents, labels, client_id=client_id
        )  # TODO: default client id

    def predict(self, emb, client_id):
        response = requests.post(
            f"{self.server.url_inf}/predictions/{MLP_MODEL_NAME}_{client_id}",
            data={"data": json.dumps({"vector": emb})},
        )
        return response.json()

    def archive_model(self, intents, labels, client_id):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            model_store = tmpdir.name if settings.storage == "S3" else MODEL_STORE
            export_path = (
                f"{tmpdir.name}/{MLP_MODEL_NAME}_{client_id}.mar"
                if settings.storage == "S3"
                else f"{model_store}/{MLP_MODEL_NAME}_{client_id}.mar"
            )
            if os.path.exists(export_path):
                os.remove(export_path)

            model_out_path = DIR_DATA / "clients" / client_id / f"{MLP_MODEL_NAME}.pt"

            with open(f"{tmpdir.name}/data.json", "w") as outfile:
                json.dump({"labels": labels.tolist(), "intents": intents}, outfile)

            subprocess.run(
                [
                    "torch-model-archiver",
                    "--model-name",
                    f"{MLP_MODEL_NAME}_{client_id}",
                    "--version",
                    "1.0",
                    "--serialized-file",
                    model_out_path,
                    "--export-path",
                    model_store,
                    "--extra-files",
                    f"{tmpdir.name}/data.json",
                    "--handler",
                    f"{DIR_HANDLERS}/handler_intent_classifier.py",
                ]
            )
            if settings.storage == "S3":
                upload_file(str(export_path))
            self.server.register_model(f"{MLP_MODEL_NAME}_{client_id}")
        finally:
            tmpdir.cleanup()
