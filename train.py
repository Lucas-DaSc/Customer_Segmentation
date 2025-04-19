# train py

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import warnings
import os
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def load_data(filepath):
    data = pd.read_csv(filepath, sep='\t')

    return data

def clean_data(data):
    # Modification age
    data["Age"] = 2021-data["Year_Birth"]
    
    #Total spendings on various items
    data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]
    
    #Deriving living situation by marital status"Alone"
    data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
    
    #Feature indicating total children living in the household
    data["Children"]=data["Kidhome"]+data["Teenhome"]
    
    #Feature for total members in the householde
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]
    
    #Feature pertaining parenthood
    data["Is_Parent"] = np.where(data.Children> 0, 1, 0)
    
    #Segmenting education 
    data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

    #For clarity
    data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

    #Droppings features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(to_drop, axis=1)

    #Outliers
    data = data[(data["Age"]<90)]
    data = data[(data["Income"]<600000)]
    
    return data

def feature_engineering(data):
    # Encoder
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)

    LE=LabelEncoder()
    for i in object_cols:
        data[i]=data[[i]].apply(LE.fit_transform)
    
    # Suppression colonnes
    ds = data.copy()
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    
    # Standar
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns)

    # PCA
    pca = PCA(n_components=3)
    pca.fit(scaled_ds)
    data_pca = pd.DataFrame(pca.transform(scaled_ds), columns=(["PC1","PC2", "PC3"]))
    
    return data_pca

def train_model(data_pca, n_clusters):
    # Log experiment dans MLflow
    mlflow.set_experiment("AgglomerativeClustering")

    with mlflow.start_run():
        model = AgglomerativeClustering(n_clusters=n_clusters)
        yhat_AC = model.fit_predict(data_pca)

        data_pca = data_pca.copy()  # éviter d'écraser l'original
        data_pca["Clusters_AC"] = yhat_AC

        # Logging du paramètre
        mlflow.log_param("n_clusters", n_clusters)

        # Évaluation
        score = silhouette_score(data_pca, model.labels_)

        # Log de la métrique
        mlflow.log_metric("silhouette_score", score)

        # CSV
        result_path = "mlflow_metrics.csv"
        df_result = pd.DataFrame([{
            "run_id": mlflow.active_run().info.run_id,
            "n_clusters": n_clusters,
            "silhouette_score": score
        }])

        # Création
        if os.path.exists(result_path):
            df_result.to_csv(result_path, mode='a', header=False, index=False)
        else:
            df_result.to_csv(result_path, index=False)


        print(f"Model trained with silhouette score: {score:.3f}")
        return model, score, data_pca

def save_model(model):
    joblib.dump(model, "model.pkl")
    return model

def run_pipeline(filepath):
    # Étape 1 : Chargement des données
    data = load_data(filepath)
    
    # Étape 2 : Nettoyage des données
    data = clean_data(data)
    
    # Étape 3 : Feature engineering
    data_pca = feature_engineering(data)
    
    # Étape 4 : Entraînement du modèle
    model, score, data_pca = train_model(data_pca, n_clusters=4)

    # Étape 5 : Sauvegarde
    model = save_model(model)

    # Afficher les résultats
    print(f'Silhouette score: {score}')

run_pipeline("marketing_campaign.csv")