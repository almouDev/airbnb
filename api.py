from flask import Flask
from flask import request
import joblib
import numpy as np
import pandas as pd
from utils import haversine,normalize
# Load the pickled model when you want to use it
model= joblib.load('knn_model.pkl') 
df=pd.read_csv("airbnb.csv")

# Use the loaded model to make predictions
# knn_from_joblib.predict(X_test) 
app = Flask(__name__)

# make the route accept json data

@app.route("/price", methods=["POST"])
def getPrice():
    # Get the data from the request, it should have same name as the columns present in airbnb.csv using
    data=request.get_json()
    hav=haversine(data["latitude"],data["longitude"],df["latitude"],df["longitude"]).mean()
    room_type=df["room_type"].value_counts().to_dict()[data["room_type"]]/len(df)
    data['state']=df["state"].value_counts().to_dict()[data["state"]]/len(df)
    data["haversine"]=hav
    data["room_type"]=room_type
    for key in data:
        data[key]=[data[key]]
    to_predict=pd.DataFrame.from_dict(data=data)
    to_predict=to_predict.drop(["longitude", "latitude"], axis=1)
    to_predict=normalize(to_predict)
    # Make the prediction using the loaded model
    prediction = model.predict(to_predict)
    # Return the prediction as a JSON response
    return {"price": int(prediction[0])}