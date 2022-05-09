"""
    intent_detection.embedding_model
    ~~~~~~~~~~~~~~~~~~~~~~~~
    EmbeddingModel class for transforming sentences into vectors
    @author: Daria Ozerova
"""
import os
import tempfile

import requests
import shutil
import json
import torch
import subprocess
import numpy as np
from typing import Union, List
from torch import is_tensor
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from intent_detection import (
    DIR_DATA,
    DIR_CLIENTS,
    DIR_MODELS,
    DIR_HANDLERS,
    MODEL_STORE,
    settings,
)
from intent_detection.services.s3 import upload_file, download_file


class EmbeddingModel(SentenceTransformer):
    """
    Model that transforms sentences into vectors
    """

    def __init__(self, transformer_name: str = None):
        """
        initialize the transformer
        :param transformer_name: name of pre-trained bi-encoder from https://www.sbert.net/docs/pretrained_models.html
        """
        self.transformer = transformer_name or settings.EMB_MODEL
        self.model_name = self.transformer
        self.parameters = {
            "batch_size": 8,
            "epochs": 4,
            "warmup_steps": 10,
            "evaluation_steps": 10,
            "output_path": "tuned_model",
        }
        self.path = str(DIR_MODELS / self.transformer)
        self.initialized = False
        # self.zip()
        # upload_file(f"{self.name}.zip")

    def initialize(self):
        path = DIR_MODELS / self.transformer

        if not self.initialized:
            if not path.exists():
                if settings.storage == "S3":
                    download_file(path=str(path))
                    shutil.unpack_archive(path, path)  # TODO: temporary
                else:
                    super().__init__(f"{self.transformer}")
                    self.save(str(path))
            super().__init__(str(path))
            self.initialized = True

    def predict(self, sentences: Union[List[str], str], **kwargs):
        """
        Transform sentences into vectors
        :param sentences: sentence or a list of sentences
        :return:
        """
        self.initialize()
        return self.encode(sentences, **kwargs)

    def pairwise_cosine_scores(self, sent1: List[str], sent2: List[str]):
        """
        Calculate cosine similarity scores for corresponding pairs of sentences
        :param sent1: list of sentences
        :param sent2: list of sentences
        :return:
        """
        self.initialize()
        emb1 = self.predict(sent1, convert_to_tensor=True)
        emb2 = self.predict(sent2, convert_to_tensor=True)

        return [util.cos_sim(e1, e2).item() for e1, e2 in zip(emb1, emb2)]

    def cosine_scores(self, sent1: List[str], sent2: Union[List[str], str]):
        """
        Calculate cosine similarity of all possible pairs from two lists of sentences
        :param sent1: list of sentences
        :param sent2: list of sentences
        :return:
        """
        self.initialize()
        emb1 = (
            sent1 if is_tensor(sent1) else self.predict(sent1, convert_to_tensor=True)
        )
        emb2 = (
            sent2 if is_tensor(sent2) else self.predict(sent2, convert_to_tensor=True)
        )

        return util.cos_sim(emb1, emb2)

    def tune(self, x_trn, y_trn, x_dev=None, y_dev=None, **kwargs):
        """
        tunes the model using labeled dataset
        :param train_data: list of sentence pairs and list of similarity scores for the training
        :param dev_data: list of sentence pairs and list of similarity scores for the validation
        :param params: training parameters: batch_size, epochs, warmup_steps, evaluation_steps, output_path
        """
        batch_size = kwargs.get("batch_size") or self.parameters.get("batch_size")
        epochs = kwargs.get("epochs") or self.parameters.get("epochs")
        warmup_steps = kwargs.get("warmup_steps") or self.parameters.get("warmup_steps")
        evaluation_steps = kwargs.get("evaluation_steps") or self.parameters.get("evaluation_steps")

        # self.initialize()
        if not x_trn:
            return

        evaluator = None

        if x_dev and y_dev and len(x_dev) == len(y_dev):
            dev = [InputExample(texts=sentences, label=score) for sentences, score in zip(x_dev, y_dev)]
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev, name="dev")

        train_examples = [InputExample(texts=pair, label=score) for pair, score in zip(x_trn, y_trn)]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(self)
        train_objectives = [(train_dataloader, train_loss)]

        self.fit(
            train_objectives,
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
            output_path=self.path,
        )
        super().__init__(self.path)

    def save_model(self):
        self.initialize()
        self.save(self.path)

    def persist_embeddings(self, samples, client_id, **kwargs):
        path = DIR_CLIENTS / client_id / f"{self.transformer}_embeddings.pt"

        if path.exists():
            embeddings = torch.load(path)
            if len(samples) == embeddings.shape[0]:
                return embeddings

        self.initialize()
        embeddings = self.encode(samples, **kwargs)

        torch.save(embeddings, path)
        return embeddings

    def encode_sentences(self, sentences, **kwargs):
        self.initialize()
        return self.encode(sentences, **kwargs)

    def zip(self):  # TODO: temporary file
        self.save_model()
        shutil.make_archive(self.transformer, "zip", self.path)


class EmbeddingModelWrapper(EmbeddingModel):
    """
    Model that transforms sentences into vectors
    """

    def __init__(self, server, transformer_name: str = None):
        """
        initialize the transformer
        :param transformer_name: a pre-trained bi-encoder from https://www.sbert.net/docs/pretrained_models.html
        """
        self.name = transformer_name or settings.EMB_MODEL
        self.server = server
        super().__init__(self.name)

        if not self.server.is_registered(self.name):
            self.archive_model()

    # TODO: cache string and list
    def predict(
        self,
        sentences: Union[List[str], str],
        convert_to_tensor=True,
        convert_to_numpy=False,
    ):
        """
        Transform sentences into vectors
        :param sentences: sentence or a list of sentences
        :param convert_to_tensor: true if embeddings need to be converted to tensor
        :return:
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        embedding = requests.post(
            f"{self.server.url_inf}/predictions/{self.name}",
            data={"data": json.dumps({"queries": sentences})},
        ).json()

        if len(embedding) == 1:
            embedding = embedding[0]  # TODO: update handler
        if convert_to_tensor:
            embedding = torch.FloatTensor(embedding)
        elif convert_to_numpy:
            embedding = np.array(embedding)
        return embedding

    def archive_model(self):
        self.save_model()

        tmpdir = tempfile.TemporaryDirectory()
        try:
            model_store = tmpdir.name if settings.storage == "S3" else MODEL_STORE
            export_path = (
                f"{tmpdir.name}/{self.name}.mar"
                if settings.storage == "S3"
                else f"{model_store}/{self.name}.mar"
            )
            if os.path.exists(export_path):
                os.remove(export_path)

            path = f"{DIR_MODELS}/{self.name}"
            extra = [
                f"{path}/{file}"
                for file in os.listdir(path)
                if "vocab." in file or "merges." in file or "config." in file
            ]

            subprocess.run(
                [
                    "torch-model-archiver",
                    "--model-name",
                    self.name,
                    "--version",
                    "1.0",
                    "--serialized-file",
                    f"{path}/pytorch_model.bin",
                    "--export-path",
                    model_store,
                    "--extra-files",
                    ",".join(extra),
                    "--handler",
                    f"{DIR_HANDLERS}/handler_embedding.py",
                ]
            )
            if settings.storage == "S3":
                upload_file(str(export_path))
            self.server.register_model(self.name, n_workers=1)
        finally:
            tmpdir.cleanup()
