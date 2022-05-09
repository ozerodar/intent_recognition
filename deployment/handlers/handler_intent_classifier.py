import json
import os
from pathlib import Path
import torch
from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.model = None
        self.intents = None
        self.labels = None
        self.device = None
        self.threshold = 0.8

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)
        self.model.eval()

        with open(model_dir + "/data.json", "r") as f:
            data = json.load(f)

        self.intents = data.get("intents")
        self.labels = data.get("labels")

        self.initialized = True

    def preprocess(self, data):
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        inputs = preprocessed_data.decode("utf-8")
        inputs = json.loads(inputs)
        emb = inputs["vector"]
        emb = torch.FloatTensor(emb)
        emb = emb.view(-1, emb.shape[0]).requires_grad_()
        return emb

    def inference(self, data, *args, **kwargs):
        outputs = torch.nn.functional.softmax(self.model.forward(data), 1)
        score, predicted = torch.max(outputs.data, 1)
        prediction = predicted[0].item()
        score = score[0].item()
        for i in range(len(self.intents)):
            if self.labels[i] == prediction:
                intent = self.intents[i] if score > self.threshold else "oos"
                return {"intent": intent, "score": score}

    def postprocess(self, inference_output):
        return [json.dumps(inference_output)]
