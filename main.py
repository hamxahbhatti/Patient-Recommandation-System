# %% Importing the libraries
from keras.models import load_model
import pickle
from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np
# %% loading the model and scaler object
def load_files():
    model = load_model('RecommandationSystem.h5')
    with open('Scaler(RecommationSystem)','rb') as file:
        scaler=pickle.load(file)
    df = pd.read_csv('stock-data.csv')
    return model,scaler,df
def predict_query(query):
    query = scaler.transform(query)
    return scaler.inverse_transform(model.predict(query))
# %% 
# app = Flask(__name__)
# app.route('/predict',methods= ['POST'])
# def receive_query():
#     print('Working')
#     request_data = request.form
#     print(request_data)
#     query = request_data['question']
#     answer = predict_query(query)
#     return jsonify({'Answer':answer})
# %% 
data=df.loc['10-26-w2019'].tolist()
np.array(data).reshape(1,-1).shape
# %% 
if __name__ == "__main__":
    model,scaler,df=load_files()
    df.set_index('Date',inplace=True)
    df.drop('Index',axis=1,inplace=True)
    data=df.loc['10-26-2019'].tolist()
    print(predict_query(np.array(data).reshape(1,-1)))

#     print(df.head())
#     app.run(port=8000)


#%%
