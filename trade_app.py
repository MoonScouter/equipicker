

import logging
from fastapi import FastAPI, APIRouter
import requests
import json
import my_func as f
import langchain

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()
api_router = APIRouter()
app.include_router(api_router, prefix="/fundamental")

@api_router.get("/business")
def business(symbol = None):

    if name is None:
        text = 'Hello!'

    else:
        text = 'Ce mai zici, ' + name + '? Maine tenis?'

    return text



