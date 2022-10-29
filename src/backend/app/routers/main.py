"""main function for uvicorn to call and create FastAPI Session
"""

from backend.app.routers.application_factory import create_app

app = create_app()