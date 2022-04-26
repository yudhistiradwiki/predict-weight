from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        model = joblib.load("model-development/weight-prediction-using-logistic-regression.pkl")
        jk = int(request.form['jk'])
        height = float(request.form['height'])
        x = np.array([[jk,height]])
        result = model.predict(x)
        output = round(result[0],2)
        return render_template('index.html', result=output)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)