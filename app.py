from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"] # Permitir todas las orígenes para desarrollo

app = FastAPI(title = "Netflix Rating Prediction API", debug=True)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y codificador
model = load(pathlib.Path('model/netflix_train_model.joblib'))

class InputData(BaseModel):
    type: str = "Movie"          # Ejemplo: 0 para 'Movie', 1 para 'TV Show'
    release_year: int = 2021   # Ejemplo: 2021
    duration: str = '110'    # Ejemplo: 110 minutos o 3 temporadas
    country: str = "United States"     # Ejemplo: 15 para 'United States'
    genre: str = "Dramas"         # Ejemplo: 7 para 'Dramas'

class OutputData(BaseModel):
    rating: str  # Ejemplo: 'TV-MA', 'R', etc.

# Cargar los LabelEncoders
le_type = load(pathlib.Path('model/label_encoder_type.joblib'))
le_country = load(pathlib.Path('model/label_encoder_country.joblib'))
le_genre = load(pathlib.Path('model/label_encoder_genre.joblib'))
le_rating = load(pathlib.Path('model/label_encoder_rating.joblib'))
model = load(pathlib.Path('model/netflix_train_model.joblib'))


@app.post('/score', response_model=OutputData)
def predict_rating(data: InputData):
    try:
        # Extraer número de duración
        if "Season" in data.duration:
            duration_num = int(data.duration.split()[0])  # Ej: "3 Seasons" → 3
        else:
            duration_num = int(data.duration.split()[0])  # Ej: "209 min" → 209

        # Codificar texto a números
        type_enc = le_type.transform([data.type])[0]
        country_enc = le_country.transform([data.country])[0]
        genre_enc = le_genre.transform([data.genre])[0]

        # Construir array de entrada
        X = np.array([[type_enc,
                       data.release_year,
                       duration_num,
                       country_enc,
                       genre_enc]])

        # Predecir rating
        pred = model.predict(X)[0]
        rating = le_rating.inverse_transform([pred])[0]

        return {
            "rating": rating
            
        }

    except Exception as e:
        return {"error": str(e)}
