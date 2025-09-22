from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import pandas as pd
import pathlib

# Cargar datos
df = pd.read_csv(pathlib.Path('data/netflix_titles.csv'))

# Filtrar columnas relevantes y eliminar nulos
df = df[['type', 'release_year', 'country', 'listed_in', 'duration', 'rating']]
df = df.dropna(subset=['type', 'release_year', 'country', 'listed_in', 'duration', 'rating'])

# Convertir duración a número (minutos o temporadas)
def extract_duration(value):
    try:
        return int(value.split(' ')[0])
    except:
        return None

df['duration_num'] = df['duration'].apply(extract_duration)
df = df.dropna(subset=['duration_num'])

# Codificar variables categóricas
le_type = LabelEncoder()
le_country = LabelEncoder()
le_genre = LabelEncoder()
le_rating = LabelEncoder()

df['type_enc'] = le_type.fit_transform(df['type'])
df['country_enc'] = le_country.fit_transform(df['country'])
df['genre_enc'] = le_genre.fit_transform(df['listed_in'])
df['rating_enc'] = le_rating.fit_transform(df['rating'])

# Definir variables de entrada y salida
X = df[['type_enc', 'release_year', 'country_enc', 'genre_enc', 'duration_num']]
y = df['rating_enc']

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluar modelo
y_pred = clf.predict(X_test)
print("Accuracy Report:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_rating.classes_))

# Guardar modelo
dump(clf, 'model/netflix_train_model.joblib')

# Guardar todos los encoders
dump(le_type, 'model/label_encoder_type.joblib')
dump(le_country, 'model/label_encoder_country.joblib')
dump(le_genre, 'model/label_encoder_genre.joblib')
dump(le_rating, 'model/label_encoder_rating.joblib')
