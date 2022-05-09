import shutil
from typing import Dict

from fastapi import APIRouter

from intent_detection.intent.intent import IntentDetection
from intent_detection.intent.utils import upload_intents
from intent_detection import DIR_CLIENTS, DEFAULT_CLIENT_ID, settings

clf = IntentDetection(torchserve_ip=settings.IP_EC2)
router = APIRouter()


@router.get("/intent")
def get_intent(query: str, client_id: str = None):
    return clf.get_intent(query, client_id=client_id)


@router.get("/clients/{client_id}")
def get_info(client_id: str):
    folder = DIR_CLIENTS / client_id
    path = folder / "intents.json"

    if path.exists():
        return {"status": "ok", "message": "client exists"}
    return {"status": "ok", "message": "client does not exist"}


@router.post("/clients/{client_id}")
def update(client_id: str, data: Dict[str, str]):
    folder = DIR_CLIENTS / client_id
    path = folder / "intents.json"

    if client_id == DEFAULT_CLIENT_ID:
        return {"status": "error", "message": "cannot update default client"}
    if len(data) < 1:
        return {"status": "error", "message": "provide more data"}
    if not path.exists():
        folder.mkdir(parents=True)

    upload_intents(path, data)
    clf.register_client(client_id)
    return {"status": "ok", "message": "registered"}


@router.delete("/clients/{client_id}")
def unregister(client_id: str):
    folder = DIR_CLIENTS / client_id

    if folder.exists() and client_id != DEFAULT_CLIENT_ID:
        shutil.rmtree(folder)
        clf.unregister_client(client_id)
        return {"status": "ok", "message": "unregistered"}
    else:
        return {"status": "error", "message": "client does not exist"}
