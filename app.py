from flask import Flask, request, render_template
import numpy as np 
import pandas
import sklearn
import pickle
# creating flask app
from flask import Flask
import pickle
import os

app = Flask(__name__)

# Define the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# importing models using relative paths
model = pickle.load(open(os.path.join(current_dir, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(current_dir, 'standscaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(current_dir, 'minmaxscaler.pkl'), 'rb'))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/crop')
def crop():
    return render_template("index.html")

@app.route('/fertilizer')
def fertilizer():
    return render_template("fertilizer.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Your existing POST handling code
        N = request.form['Nitrogen']
        P = request.form['Phosporus']
        K = request.form['Potassium']
        temp = request.form['Temperature']
        humidity = request.form['Humidity']
        ph = request.form['Ph']
        rainfall = request.form['Rainfall']
        
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)
        
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
        
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        
        return render_template('index.html', result=result)
    return render_template('index.html')

@app.route("/predict_fertilizer", methods=['GET', 'POST'])
def predict_fertilizer():
    if request.method == 'POST':
        # Get values from the form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])

        # Make prediction
        features = np.array([[N, P, K]])
        prediction = fertilizer_model.predict(features)

        fertilizer_dict = {
            0: "TEN-TWENTY SIX-TWENTY SIX",
            1: "Fourteen-Thirty Five-Fourteen",
            2: "Seventeen-Seventeen-Seventeen",
            3: "TWENTY-TWENTY",
            4: "TWENTY EIGHT-TWENTY EIGHT",
            5: "DAP",
            6: "UREA"
        }

        result = fertilizer_dict.get(prediction[0], "Could not determine the appropriate fertilizer")
        return render_template('fertilizer.html', result=result)
    return render_template('fertilizer.html')

if __name__ == "__main__":
    app.run(debug=True)