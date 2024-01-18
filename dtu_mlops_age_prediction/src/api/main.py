import sys
sys.path.append('./src')
from fastapi import FastAPI
from http import HTTPStatus
from fastapi import UploadFile, File
from typing import Optional
from torchvision.transforms import transforms
from predict_model import predict
import uvicorn
import os

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
async def cv_model(data: UploadFile = File(...)):
    with open('image.png', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()
        label = predict('models/model.pt',"image.png")
   
    return {'Age predicted': label}


uvicorn.run(app, port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")