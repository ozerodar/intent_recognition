from pathlib import Path
from pydantic import BaseSettings, SecretStr, Field


class Settings(BaseSettings):
    aws_key: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    region: str = Field(..., env="AWS_REGION")
    IP_EC2: str = Field(..., env="EC2_IP")
    PORT_INF: str = Field(..., env="PORT_INFERENCE")
    PORT_MNG: str = Field(..., env="PORT_MANAGEMENT")
    EMB_MODEL: str = Field(..., env="EMB_MODEL")
    S3_BUCKET: str = Field(..., env="S3_BUCKET")
    storage: str = Field(..., env="STORAGE")

    DEFAULT_PROJECT_ID: str = Field(..., env="DEFAULT_PROJECT_ID")
    DEFAULT_TOKEN: str = Field(..., env="DEFAULT_TOKEN")
    IP: str = Field(..., env="IP")
    PORT: str = Field(..., env="PORT")

    class Config:
        case_sentive = False
        env_file = Path(__file__).parent.parent / ".env.local"


MODEL_OBJECT_NAME = "intent-classifier.pkl"
MLP_MODEL_NAME = "intent-classifier"

DIR_DATA = Path(__file__).parent.parent / "data"
DIR_CLIENTS = Path(__file__).parent.parent / "data" / "clients"
DIR_DEPLOYMENT = Path(__file__).parent.parent / "deployment"
DIR_MODELS = Path(__file__).parent.parent / "data" / "models"
MODEL_STORE = Path(__file__).parent.parent / "model_store"
EXPERIMENTS_DIR_DATA = Path(__file__).parent.parent / "experiments" / "data"
DIR_HANDLERS = Path(__file__).parent.parent / "deployment" / "handlers"
DEFAULT_CLIENT_ID = "0"

if not MODEL_STORE.exists():
    MODEL_STORE.mkdir(parents=True)

settings = Settings()
AWS_ACCESS_KEY_ID = settings.aws_key
AWS_SECRET_ACCESS_KEY = settings.aws_access_key
AWS_REGION = settings.region
S3_BUCKET = settings.S3_BUCKET
