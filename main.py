import io
import pickle
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware

with open('model.pkl','rb') as f:
  model = pickle.load(f)

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.post('/predict-image/')
async def predict_image(file: UploadFile = File(...)):
  contents = await file.read()
  pil_image = Image.open(io.BytesIO(contents)).convert('L')
  pil_image = ImageOps.invert(pil_image)
  pil_image = pil_image.resize((28,28),Image.Resampling.NEAREST)
  img_array = np.array(pil_image).reshape(1,-1)
  predictions = model.predict(img_array)
  return {'prediction': int(predictions[0])}