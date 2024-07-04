import numpy as np
import polars as pl
from meteostat import Point, Daily
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import ray
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# lokacija in časovno obdobje
start = datetime(2023, 6, 1)
end = datetime(2024, 7, 4)
location = Point(46.55, 15.495)  # Maribor, Slovenija

# dobivanje podatkov
data = Daily(location, start, end)
data = data.fetch()

# pretvorba v df
df = pl.DataFrame(data)

# preimenovanje
df = df.rename({
    'tavg': 'temperature',
    'tmin': 'min_temperature',
    'tmax': 'max_temperature',
    'prcp': 'precipitation',
    'wdir': 'wind_direction',
    'wspd': 'wind_speed',
    'pres': 'pressure'
})

# filtriranje po uporabnosti
df = df.select([
    'temperature',
    'min_temperature',
    'max_temperature',
    'precipitation',
    'wind_direction',
    'wind_speed',
    'pressure'
])

# dodajanje target za 7 dni
forecast_days = 7
for i in range(1, forecast_days + 1):
    df = df.with_columns([
        (pl.col('temperature').shift(-i)).alias(f'target_avg_day_{i}'),
        (pl.col('min_temperature').shift(-i)).alias(f'target_min_day_{i}'),
        (pl.col('max_temperature').shift(-i)).alias(f'target_max_day_{i}')
    ]).drop_nulls()

# normalizacija
df = df.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_norm") 
    for col in df.columns if not col.startswith('target')
])

# dodatni targeti
df = df.with_columns([
    (pl.col("temperature_norm") + pl.col("pressure_norm")).alias("temp_pressure_sum"),
    (pl.col("wind_speed_norm") + pl.col("wind_direction_norm")).alias("wind_speed_direction_sum")
])

# zmanjšanje dimenzionalnosti
features = [col for col in df.columns if col.endswith("_norm") or col in ["temp_pressure_sum", "wind_speed_direction_sum"]]
X = df.select(features).to_numpy()
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)

for i in range(X_reduced.shape[1]):
    df = df.with_columns(pl.Series(f"PC{i+1}", X_reduced[:, i]))

# izbira najpomembnejših targetov
ray.init(ignore_reinit_error=True)

@ray.remote
def find_top_features(data):
    return np.argsort(np.var(data, axis=0))[-10:]

result_id = find_top_features.remote(X_reduced)
top_features = ray.get(result_id)

# treniranje in napovedovanje
def train_and_predict(df, targets, forecast_days, top_features):
    models = []
    predictions = {target: [] for target in targets}
    
    for day in range(1, forecast_days + 1):
        for target in targets:
            y = df[f"{target}_day_{day}"].to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X_reduced[:, top_features], y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            
            with mlflow.start_run():
                model.fit(X_train, y_train)
                mlflow.sklearn.log_model(model, f"{target}_model_day_{day}")
                mlflow.log_metric(f"{target}_train_score_day_{day}", model.score(X_train, y_train))
                mlflow.log_metric(f"{target}_test_score_day_{day}", model.score(X_test, y_test))
            
            models.append((model, target, day))
    
    # napovedovanje
    future_features = df.select(features).tail(1).to_numpy()
    future_pca = pca.transform(future_features)
    
    for model, target, day in models:
        predictions[target].append(model.predict(future_pca[:, top_features])[0])
    
    return predictions

# treniranje in napoved povprečnih, minimalnih in maksimalnih temperatur
targets = ['target_avg', 'target_min', 'target_max']
predictions = train_and_predict(df, targets, forecast_days, top_features)

# vizualiacija
future_dates = [end + timedelta(days=i) for i in range(1, forecast_days + 1)]
plt.figure(figsize=(10, 6))

for target, label, color in zip(targets, ["Povprečna", "Minimalna", "Maksimalna"], ['g', 'b', 'r']):
    plt.plot(future_dates, predictions[target], marker='o', linestyle='-', color=color, label=f'Napovedana {label} temperatura')

plt.xlabel('Datum')
plt.ylabel('Napovedana temperatura (°C)')
plt.title('Napovedane temperature za naslednjih 7 dni')
plt.grid(True)
plt.legend()
plt.savefig("forecast_all.png")
plt.show()

# izpis napovedi
forecast_df = pl.DataFrame({
    "Datum": future_dates,
    "Napovedana povprečna temperatura": predictions['target_avg'],
    "Napovedana minimalna temperatura": predictions['target_min'],
    "Napovedana maksimalna temperatura": predictions['target_max']
})
print("Napovedane temperature za naslednjih 7 dni:")
print(forecast_df)