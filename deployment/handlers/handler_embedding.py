"""
handler_embedding.py
"""
from sentence_transformers import SentenceTransformer
import json
import zipfile
from ts.torch_handler.base_handler import BaseHandler


class SentenceTransformerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        with zipfile.ZipFile(model_dir + "/pytorch_model.bin", "r") as zip_ref:
            zip_ref.extractall(model_dir)
        self.model = SentenceTransformer(str(model_dir))

    def preprocess(self, data):
        return json.loads(data[0].get("data").decode("utf-8"))["queries"]

    def inference(self, data, *args, **kwargs):
        return self.model.encode(data)

    def postprocess(self, data):
        return [json.dumps(data.tolist())]
