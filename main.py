
import logging
from fastapi import FastAPI, APIRouter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()
my_first_api_router = APIRouter()

@my_first_api_router.get("/hello")
def hello(name = None):

    if name is None:
        text = 'Hello!'

    else:
        text = 'Ce mai zici, ' + name + '? Maine tenis?'

    return text


app.include_router(my_first_api_router, prefix="/my-first-api")




