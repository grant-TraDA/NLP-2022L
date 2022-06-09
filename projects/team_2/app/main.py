from fastapi import FastAPI
from mangum import Mangum
from starlette.middleware.cors import CORSMiddleware

from api.files import router

app = FastAPI(title='Serverless Lambda FastAPI')

ALLOWED_HOSTS = ("http://127.0.0.1:8080,http://localhost:8080,http://0.0.0.0:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os 
if "app" in os.listdir():
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
elif "static" in os.listdir():
    app.mount("/static", StaticFiles(directory="static"), name="static")

if "app" in os.listdir():
    app.mount("/scripts", StaticFiles(directory="app/scripts"), name="scripts")
elif "static" in os.listdir():
    app.mount("/scripts", StaticFiles(directory="scripts"), name="scripts")

app.include_router(router, prefix='')

# to make it work with Amazon Lambda, we create a handler object
handler = Mangum(app=app)
