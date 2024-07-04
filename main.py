import numpy as np
import polars as pl
from meteostat import Point, Daily
from datetime import datetime
from sklearn.decomposition import PCA
import ray
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Lokacija in časovno obdobje
start = datetime(2023, 6, 1)
end = datetime(2024, 7, 4)
location = Point(46.55, 15.495)  # Maribor, Slovenija

# Pridobivanje podatkov
data = Daily(location, start, end)
data = data.fetch()

# Pretvorba v Polars DataFrame
df = pl.DataFrame(data)

# Preimenovanje stolpcev za boljšo razumljivost
df = df.rename({
    'tavg': 'temperature',
    'tmin': 'min_temperature',
    'tmax': 'max_temperature',
    'prcp': 'precipitation',
    'wdir': 'wind_direction',
    'wspd': 'wind_speed',
    'pres': 'pressure'
})

# Filtriranje in uporaba le potrebnih stolpcev
df = df.select([
    'temperature',
    'min_temperature',
    'max_temperature',
    'precipitation',
    'wind_direction',
    'wind_speed',
    'pressure'
])

# Dodajanje ciljne spremenljivke
df = df.with_columns((pl.col('temperature').shift(-1)).alias('target')).drop_nulls()

# Izpis osnovnih statističnih podatkov
print("Osnovne statistične meritve za originalne značilke:")
print(df.describe())

# Normalizacija podatkov
df = df.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_norm") 
    for col in df.columns if col != "target"
])

# Izpis osnovnih statističnih podatkov za normalizirane značilke
print("\nOsnovne statistične meritve za normalizirane značilke:")
print(df.select([col for col in df.columns if col.endswith("_norm")]).describe())

# Ustvarjanje dodatnih značilk
df = df.with_columns([
    (pl.col("temperature_norm") + pl.col("pressure_norm")).alias("temp_pressure_sum"),
    (pl.col("wind_speed_norm") + pl.col("wind_direction_norm")).alias("wind_speed_direction_sum")
])

# Zmanjšanje dimenzionalnosti
features = [col for col in df.columns if col.endswith("_norm") or col in ["temp_pressure_sum", "wind_speed_direction_sum"]]
X = df.select(features).to_numpy()
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)

for i in range(X_reduced.shape[1]):
    df = df.with_columns(pl.Series(f"PC{i+1}", X_reduced[:, i]))

# Izbira najpomembnejših značilk
ray.init(ignore_reinit_error=True)

@ray.remote
def find_top_features(data):
    return np.argsort(np.var(data, axis=0))[-10:]

result_id = find_top_features.remote(X_reduced)
top_features = ray.get(result_id)

# Sledenje eksperimentom in treniranje modela
y = df["target"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_reduced[:, top_features], y, test_size=0.2, random_state=42)
model = RandomForestRegressor()

with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("train_score", model.score(X_train, y_train))
    mlflow.log_metric("test_score", model.score(X_test, y_test))

# Napovedi modela
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Vizualizacija rezultatov
plt.figure(figsize=(14, 7))

# Scatter plot za napovedi proti dejanskim vrednostim na testnem sklopu
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([-10, 40], [-10, 40], 'r--')
plt.xlabel('Dejanske vrednosti')
plt.ylabel('Napovedane vrednosti')
plt.title('Napovedi proti dejanskim vrednostim (testni sklop)')

# Histogram napak na testnem sklopu
plt.subplot(1, 2, 2)
errors = y_test - y_pred_test
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel('Napaka')
plt.ylabel('Frekvenca')
plt.title('Porazdelitev napak (testni sklop)')

plt.tight_layout()
plt.savefig("model_evaluation.png")
plt.show()

# Podrobna vizualizacija rezultatov
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label="Dejanske vrednosti", alpha=0.7)
plt.plot(range(len(y_test)), y_pred_test, label="Napovedane vrednosti", alpha=0.7)
plt.xlabel('Vzorec')
plt.ylabel('Temperatura')
plt.title('Napovedane proti dejanskim vrednostim skozi čas')
plt.legend()
plt.savefig("vizualizacija.png")
plt.show()

# Korelacijska matrika
corr_matrix = df.select([col for col in df.columns if col != "target"]).to_pandas().corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelacijska matrika značilk')
plt.show()

# Izpis vrednosti napovedi in napak za vzorce iz testnega sklopa
results = pl.DataFrame({
    "Dejanske vrednosti": y_test,
    "Napovedane vrednosti": y_pred_test,
    "Napake": y_test - y_pred_test
})
print("Podrobni rezultati modela za testni sklop:")
print(results.head(10))
