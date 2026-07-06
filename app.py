import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "kmeans_riesgo_actuarial.pkl"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
CLUSTERS_CSV_PATH = BASE_DIR / "outputs" / "insurance_con_clusters.csv"

REGIONES = ["southeast", "southwest", "northeast", "northwest"]

EXPLICACIONES = {
    "Bajo": "Este cliente fue agrupado con perfiles de menor costo medico promedio.",
    "Medio": "Este cliente fue agrupado con perfiles de costo y factores de riesgo intermedios.",
    "Alto": "Este cliente fue agrupado con perfiles de mayor costo medico promedio y/o factores de riesgo relevantes.",
}


def load_model():
    modelo = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    mapa_riesgo = {int(k): v for k, v in metadata["mapa_riesgo"].items()}
    return modelo, metadata, mapa_riesgo


def load_clusters_csv():
    try:
        return pd.read_csv(CLUSTERS_CSV_PATH)
    except FileNotFoundError:
        return None


def evaluar_cliente(modelo, mapa_riesgo, age, sex, bmi, children, smoker, region, charges):
    cliente = pd.DataFrame([{
        "age": age,
        "sex": str(sex).lower(),
        "bmi": bmi,
        "children": children,
        "smoker": str(smoker).lower(),
        "region": str(region).lower(),
        "charges": charges,
    }])

    cluster = int(modelo.predict(cliente)[0])
    riesgo = mapa_riesgo[cluster]

    return cluster, riesgo, EXPLICACIONES[riesgo]


# ---------- Pagina ----------

st.title("Clasificador SVM Seguros")
st.write("Anny Yohana Gutierrez - Cuenta: 20211930078")

st.write(
    "Esta aplicacion clasifica el perfil de un cliente de seguro medico en un "
    "nivel de riesgo (Bajo, Medio o Alto), usando un modelo de clustering "
    "(K-means) entrenado con el dataset insurance.csv."
)

modelo, metadata, mapa_riesgo = load_model()
df_clusters = load_clusters_csv()

st.write("Numero de clusters del modelo:", metadata["n_clusters"])
st.write("Silhouette score:", metadata["silhouette_score"])

st.header("Datos del cliente")

age = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1)
sex = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("BMI (indice de masa corporal)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Numero de hijos", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Fumador", ["yes", "no"])
region = st.selectbox("Region", REGIONES)
charges = st.number_input("Cargos medicos (charges, en USD)", min_value=0.0, value=10000.0, step=100.0)

if st.button("Evaluar riesgo"):
    cluster, riesgo, explicacion = evaluar_cliente(
        modelo, mapa_riesgo, age, sex, bmi, children, smoker, region, charges
    )

    st.header("Resultado")
    st.write("Riesgo actuarial:", riesgo)
    st.write("Cluster asignado:", cluster)
    st.write(explicacion)

    if df_clusters is not None:
        st.header("Comparacion con la cartera de clientes")

        resumen = df_clusters.groupby("riesgo_actuarial").agg(
            cantidad_clientes=("riesgo_actuarial", "count"),
            cargos_promedio=("charges", "mean"),
            edad_promedio=("age", "mean"),
            bmi_promedio=("bmi", "mean"),
        ).round(2)

        st.write(resumen)

        st.write("Cargos medicos promedio por nivel de riesgo:")
        st.bar_chart(resumen["cargos_promedio"])

        st.write(
            f"El cliente evaluado tiene charges = {charges:,.2f}, frente a un "
            f"promedio de {resumen.loc[riesgo, 'cargos_promedio']:,.2f} en su "
            f"grupo de riesgo ({riesgo})."
        )
