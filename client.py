import requests
# Datos completos para el título "I, Tonya"
body = {
    "title": "I, Tonya",
    "description": "Competitive ice skater Tonya Harding rises amidst scandal.",
    "type": "Movie",
    "release_year": 2017,
    "duration": "121 min",
    "country": "United States",
    "genre": "Dramas"
}

# Enviar POST request al endpoint de predicción
response = requests.post(url='http://127.0.0.1:8000/score', json=body)

# Mostrar resultado
print(response.json())
