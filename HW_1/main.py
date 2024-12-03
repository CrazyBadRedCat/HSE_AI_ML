
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import io

with open("best_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def preprocess_item(item: Item) -> dict:
    """
    Преобразует объект Item в формат, подходящий для модели.
    """
    return {
        "name": item.name,
        "year": item.year,
        "selling_price": item.selling_price,
        "km_driven": item.km_driven,
        "fuel": item.fuel,
        "seller_type": item.seller_type,
        "transmission": item.transmission,
        "owner": item.owner,
        "mileage": item.mileage,
        "engine": item.engine,
        "max_power": item.max_power,
        "torque": item.torque,
        "seats": item.seats,
    }

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Предсказывает стоимость машины для одного объекта.
    """
    try:
        data = pd.DataFrame([preprocess_item(item)])
        prediction = model.predict(data)
        return prediction[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_items")
def predict_items(file: UploadFile):
    """
    Принимает CSV-файл с признаками объектов и возвращает CSV-файл с предсказаниями.
    """
    try:
        contents = file.file.read()
        data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        required_columns = [
            "name", "year", "km_driven", "fuel",
            "seller_type", "transmission", "owner", "mileage",
            "engine", "max_power", "seats"
        ]
        if not all(column in data.columns for column in required_columns):
            raise HTTPException(status_code=400, detail="Некорректный формат файла. Проверьте столбцы.")

        predictions = model.predict(data)

        data["predicted_price"] = predictions

        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0)

        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
