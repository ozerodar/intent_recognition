from fastapi import FastAPI
from intent_detection.api import api_router


def create_app():
    application = FastAPI(title="Intent Detection", docs_url="/")
    application.include_router(api_router)

    @application.on_event("startup")
    async def app_startup():  # pylint: disable=unused-variable
        print("started")

    return application


app = create_app()
