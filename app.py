from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline import predict_pipeline


flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return render_template('index.html')

@flask_app.route('/predictcharge',methods=['GET', 'POST'])
def predict_charges():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        data = predict_pipeline.CustomData(
            age = int(request.form.get('age')),
            sex = request.form.get('sex'),
            bmi = float(request.form.get('bmi')),
            children = int(request.form.get('children')),
            smoker = request.form.get('smoker'),
            region = request.form.get('region')
        )

        pred_df = data.get_data_as_dataframe()

        print(pred_df)

        prediction_obj = predict_pipeline.PredictPipeline()
        results = prediction_obj.predict(pred_df)

        return render_template('home.html', results=results[0])
        


if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', debug = True)


