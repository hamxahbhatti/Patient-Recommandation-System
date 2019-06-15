from flask import Flask
from flask_restful import Resource, Api
from keras.models import load_model
from keras import backend as K
import pickle
from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np

app = Flask(__name__)

def load_files():
    model = load_model('RecommandationSystem.h5')
    with open('Scaler(RecommationSystem)','rb') as file:
        scaler=pickle.load(file)
    df = pd.read_csv('.csv')  #############################Type your file name here i changed it
    return model,scaler,df

def predict_query(query,model,scaler):
    query = scaler.transform(query)
    return scaler.inverse_transform(model.predict(query))

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/data/<prevmonth>")
def hello_there(prevmonth):
        model,scaler,df=load_files()
        df.set_index('Month',inplace=True)
        data=df.loc[prevmonth].tolist()
        data=predict_query(np.array(data).reshape(1,-1),model,scaler).tolist()
        K.clear_session()
        return jsonify(result=data)
if __name__ == "__main__":

    app.debug = True
    app.run(port=5000)
