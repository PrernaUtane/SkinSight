from app.config import HOST, PORT
from app.main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, workers=1)
