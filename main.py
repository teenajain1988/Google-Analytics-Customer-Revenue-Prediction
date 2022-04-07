
import numpy as np
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/",methods=["GET"])
def home_page():
    return render_template("home.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        df = pd.read_csv("https://raw.githubusercontent.com/teenajain1988/Google_analytics-finale-submission-file/main/lgb_models2.csv")
        df = df.astype({"fullVisitorId": str})
        fullVisitorId=str(request.form["fullVisitorId"])
        if fullVisitorId in df["fullVisitorId"].values:
            result = df[df["fullVisitorId"] == fullVisitorId]["PredictedLogRevenue"].values[0]
            return render_template("home1.html",result=f"The predicted log revenue of of the person having fullVisitorId  {fullVisitorId} is {result}")
        else:
            return render_template("home2.html")
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run()