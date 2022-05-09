import torch

from intent_detection.intent.embedding import EmbeddingModelWrapper
from intent_detection.intent.mlp_classifier import MlpClassifierWrapper
from intent_detection.intent.utils import load_embeddings, get_intents
from intent_detection.services.torchserve_management import TorchServer
from intent_detection import DIR_CLIENTS, MLP_MODEL_NAME, DEFAULT_CLIENT_ID, settings


class IntentDetection:
    def __init__(self, torchserve_ip, emb_model_name=None):
        emb_model_name = emb_model_name or settings.EMB_MODEL
        self.intent_clf = None
        self.nn_clf = None
        self.server = TorchServer(torchserve_ip)
        self.emb_model = EmbeddingModelWrapper(self.server, emb_model_name)
        self.intent_clf = MlpClassifierWrapper(self.server)
        self.register_client(DEFAULT_CLIENT_ID)

    def get_intent_cosine_scores(self, query, client_id):
        templates, intents, labels = get_intents(client_id)
        if not (embeddings := load_embeddings(client_id, self.emb_model.name)):
            embeddings = templates

        cosine_scores = self.emb_model.cosine_scores(embeddings, query)
        match_idx = torch.argmax(cosine_scores)
        max_score = cosine_scores[match_idx][0].item()
        return {
            "matching sentence": templates[match_idx],
            "score": max_score,
            "intent": intents[match_idx],
        }

    def get_intent_supervised(self, query, client_id):
        emb = self.emb_model.predict(query, convert_to_tensor=False)
        return self.intent_clf.predict(emb, client_id=client_id)

    def get_intent(self, query, client_id=None):
        client_id = client_id or DEFAULT_CLIENT_ID
        if not self.server or self.server.is_registered(
            f"{MLP_MODEL_NAME}_{client_id}"
        ):
            # result_unsupervised = self.get_intent_cosine_scores(query, client_id)
            result_supervised = self.get_intent_supervised(query, client_id)
            return result_supervised
            # return {"Unsupervised": result_unsupervised, "Supervised": result_supervised}
        else:
            return {"ERROR": f"Model {MLP_MODEL_NAME}_{client_id} is not registered"}

    def register_client(self, client_id):
        templates, intents, labels = get_intents(client_id)
        embeddings = self.emb_model.persist_embeddings(
            templates, convert_to_tensor=True, client_id=client_id
        )

        self.intent_clf.register_model(
            client_id=client_id,
            embeddings=embeddings,
            intents=intents,
            labels=labels,
            verbose=True,
            l_rate=0.0001,
        )  # TODO: fix number of epoch

    def unregister_client(self, client_id):
        self.server.unregister_model(model_name=f"{MLP_MODEL_NAME}_{client_id}")
