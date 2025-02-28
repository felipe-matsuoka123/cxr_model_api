from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image
from model.model import predict

app = FastAPI()

class ImageRequest(BaseModel):
    url: str

@app.post("/predict/")
async def predict_image(request: ImageRequest):
    try:
        # Download image from URL
        response = requests.get(request.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Error downloading image")

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Run prediction
        result = predict(image)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))