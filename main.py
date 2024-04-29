import numpy as np
import pandas as pd
from utils import *
import joblib
def test(df):
    y_train=df["price"]
    x_train=df.drop(["price"],axis=1)
    df['price']=y_train
    knn=get_knn_instance()
    knn.fit(x_train,y_train)
    print('RMSE:',get_cross_validation_score(x_train,y_train,knn,10))

def getSectedFeatures(x_train,y_train,methode="SelectKBest"):
    features_selected=[]
    scores=[]
    for nfeatures in range(3,len(x_train.columns)+1):
        features=feature_selection(x_train,y_train,methode=methode,n_features=nfeatures)
        x_train1 = df[features]
        knn=get_knn_instance()
        knn.fit(x_train1,y_train)
        features_selected.append(features)
        scores.append(knn.score(x_train1,y_train))
    return features_selected[scores.index(max(scores))]
#load data using pandas, the file is named airbnb.csv
df = pd.read_csv("airbnb.csv")
print("---Donnees chargees---")
#1 Suppression des colonnes non pertinantes
df=delete_columns(df)
print("Colonnes non pertinantes supprimes")
df.info()
# Melanger les lignes
df=randomize(df)
#nettoyages de la colonnes price de telle sorte a avoir des valeurs numeriques(ne contenant le signe $)
df=cleanup_price_column(df)

#2 Les colonnes non numeriques
non_numeric_columns = get_non_numeric_columns(df)
non_numeric_columns.append("zipcode")
#Transformation  des colonnes non ordinals et non numeriques
df=generate_ordinal_columns(df,non_numeric_columns)
print("Colonnes non ordinales et non numeriques transformees")
df.info()
# 3 Feature engeneering
# utilise fonction harversine pour integrer la longitude et la latitude dans le dataframe
df["haversine"] = df.apply(lambda row: haversine(row["latitude"], row["longitude"],df["latitude"],df["longitude"]).mean(), axis=1)
df = df.drop(["longitude", "latitude"], axis=1)
df.info()
# 4 Geger les valeurs manquantes

critical_columns = ['bedrooms', 'bathrooms', 'beds']
df=df.dropna(subset=critical_columns)
df.fillna(df.mean(),inplace=True)
print('Premier test du modèle')
test(df)
#Entraine le modele et test le grace a la fonction get_cross_validation_score

#5 Normalisation ou standardisation
# Normalisation
df = normalize(df)
print("Resultat apres normalisation")
test(df)

# 6 Identification et suppression des valeurs extremes qui peuvent fausser le modele
df=remove_extreme_values(df)
print("Resultat apres suppression des valeurs extremes")
test(df)
#7 Feature selection 
y_train = df["price"]
x_train = df.drop(["price"],axis=1)
features=getSectedFeatures(x_train,y_train,methode="SelectKBest")
print("Les caractérisques qui optimisent la performance du modele sont:")
print(features)
x_train=x_train[features]
#8 Modelisation
knn=get_knn_instance()
knn.fit(x_train,y_train)
print("Resultat du modèle avec les meilleurs caractéristiques")
print("RMSE:",get_cross_validation_score(x_train,y_train,knn,10))
knn=get_knn_instance()
best_params=doRandomSearch(x_train,y_train,knn)
print('Best hyper params ')
print(best_params)
best_params["algorithm"]="auto"
knn=knn.set_params(**best_params)
knn.fit(x_train,y_train)
print("Resultat du modèle avec les meilleurs hyperparametres")
print("RMSE:",get_cross_validation_score(x_train,y_train,knn,10))
# Save the trained model as a pickle string.
joblib.dump(knn, 'knn_model.pkl')



