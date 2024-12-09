from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import json
import os

class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


import pickle
with open('model_properties.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
model = loaded_model['model']
scaler = loaded_model['scaler']
encoder = loaded_model['encoder']


app = FastAPI()

def preprocess_data(item: Item) -> np.ndarray:

    data = pd.DataFrame([json.loads(item.model_dump_json())])
    data['engine'] = data['engine'].str.extract(r'([\d.]+)').astype(int)
    data['mileage'] = data['mileage'].str.extract(r'([\d.]+)').astype(float)
    data['max_power'] = data['max_power'].str.extract(r'([\d.]+)').astype(float)
    data['seats'] = data['seats'].astype(int)
    data_encoded = OHE.transform(data[['fuel', 'seller_type', 'transmission', 'owner','seats']])
    numerical_data = data[['year','mileage','engine','max_power', 'km_driven']]
    processed_data = np.hstack((numerical_data, data_encoded.toarray()))
    return processed_data


async def predict_item(item: Item) -> float:
    features = preprocess_data(item)
    prediction = model.predict(features)[0]
    return float(prediction)

def predict_items(items: List[Item]) -> List[float]:
    predictions = []
    for item in items:
        features = preprocess_data(item)
        prediction = model.predict(features)[0]
        predictions.append(float(prediction))
    return predictions

@app.post("/predict_item/")
async def predict(item: Item):
    prediction = await predict_item(item)

    return {
        "prediction": prediction
    }

@app.post("/predict_items/")
async def predict_csv(file: UploadFile = File(...)):
    # Загрузка CSV-файла н
    items = pd.read_csv(file.file)

    # Преобразование DataFrame в список
    items_list = [Item(**row) for index, row in items.iterrows()]
    predictions = predict_items(items_list)
    items['predicted_price'] = predictions
    output_file = 'predicted_items.csv'
    items.to_csv(output_file, index=False)
    return FileResponse(output_file, media_type='text/csv', filename='predicted_items.csv')
