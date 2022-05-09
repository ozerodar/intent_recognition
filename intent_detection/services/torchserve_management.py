import requests

from intent_detection.services import s3
from intent_detection import settings


class TorchServer:
    def __init__(self, ip=None):
        ip = ip or settings.IP_EC2
        self.url_mng = f"{ip}:{settings.PORT_MNG}"
        self.url_inf = f"{ip}:{settings.PORT_INF}"

    def unregister_model(self, model_name):
        response = requests.delete(f"{self.url_mng}/models/{model_name}")
        if response.status_code == 200:
            print(f"Model {model_name} is successfully unregistered")
        else:
            print(f"Cannot unregister model {model_name}. Response: {response.json()}")

    def register_model(self, model_name, n_workers=1):
        if self.is_registered(model_name):
            self.unregister_model(model_name)

        if settings.storage != "S3":
            url = f"{model_name}.mar"
        else:
            url = s3.generate_url(f"{model_name}.mar")
        response = requests.post(url=f"{self.url_mng}/models?url={url}")
        print(response.json())

        self.add_workers(model_name, n_workers)

    def is_registered(self, model_name):
        response = requests.get(f"{self.url_mng}/models/")

        models = response.json()["models"]
        for model in models:
            if model["modelName"] == model_name:
                return True
        return False

    def add_workers(self, model_name, n_workers):
        if not self.is_registered(model_name):
            print(f"Model {model_name} is not registered")
            return

        response = requests.put(
            url=f"{self.url_mng}/models/{model_name}?min_worker={n_workers}"
        )
        print(response.json())

    def unregister_all(self):
        response = requests.get(f"{self.url_mng}/models/")
        models = response.json()["models"]

        for model in models:
            self.unregister_model(model["modelName"])


if __name__ == "__main__":
    server = TorchServer()
    server.unregister_all()
    # server.unregister_model("intent-classifier_0")
