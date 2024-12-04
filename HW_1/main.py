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

def mileage_preprocess(x):
    if x is None:
        return np.nan

    x = str(x).replace(" kmpl", "")
    if x.endswith("km/kg") or (x == ""):
        return np.nan
    else:
        return float(x)

def engine_preprocess(x):
    if x is None:
        return np.nan

    x = str(x).replace(" CC", "")
    if x == "":
        return np.nan
    else:
        return float(x)

def max_power_preprocess(x):
    if x is None:
        return np.nan
    
    x = str(x).replace(" bhp", "")
    if x == "":
        return np.nan
    else:
        return float(x)

def name_preprocess(x):
    return str(x).split()[0]

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует DataFrame в формат, подходящий для модели.
    """
    df = df.drop(columns=["torque", "selling_price"], errors = "ignore")
    
    df["mileage"] = df["mileage"].apply(mileage_preprocess)
    df["engine"] = df["engine"].apply(engine_preprocess)
    df["max_power"] = df["max_power"].apply(max_power_preprocess)
    df["name"] = df["name"].apply(name_preprocess)
    
    df = df.fillna({
        "mileage": 19.30,
        "engine": 1248.00,
        "seats": 5.00,
        "max_power": 81.86,
    })
    
    df = df.astype({
        "mileage": float,
        "engine": int,
        "seats": int,
        "max_power": float,
    })
    
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Предсказывает стоимость машины для одного объекта.
    """
    try:
        data = pd.DataFrame([item.model_dump()])
        data = preprocess_df(data)
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

        predictions = model.predict(preprocess_df(data.copy()))

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
