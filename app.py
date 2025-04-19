import streamlit as st
import pandas as pd
import os

st.title("Dashboard MLOps")

# Résumé des métriques MLflow
st.subheader("Dernières métriques du modèle")
if os.path.exists("mlflow_metrics.csv"):
    metrics = pd.read_csv("mlflow_metrics.csv")
    st.dataframe(metrics.tail(5))
else:
    st.warning("Pas encore de métriques enregistrées.")