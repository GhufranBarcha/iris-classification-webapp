from flask import Flask, render_template ,url_for ,request
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

app = Flask(__name__)

## Import the models
pipeline = joblib.load("irisModelPipeline.pkl")
encoder = joblib.load("irisEncoderPipeline.pkl")
print(encoder.classes_)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit" ,methods = ['POST','GET'])
def submit():
    

    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        data = {
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
          }

        df1= pd.DataFrame(data)
 
        

        print(df1)
        
        # Predict the data
        transformed_data = pipeline["scaling"].transform(df1)
        print(transformed_data)
        prediction = pipeline["estimator"].predict(transformed_data)

        print("The Label is:", encoder.classes_[prediction][0])

    return render_template("index.html" ,predicted = f"The Label is {encoder.classes_[prediction][0]}")

if __name__ == "__main__":
    app.run(port=3000, debug=True)