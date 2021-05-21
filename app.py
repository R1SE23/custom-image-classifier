from flask import Flask, request, render_template, jsonify, send_from_directory, current_app, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from botnoi import cv
import requests
import joblib
import numpy as np
import os
from trained_model_from_google_image_search import run

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict_url')
def classifier():
    try:
        img_url = request.values['p_image_url'] 
        # resize to 224x224x3
        img = cv.image(img_url)
        # extract features
        feat = img.getresnet50()
        # load trained model
        model = joblib.load('vehicle-classification.p')
        
        #Predict classes and probabilities with LinearSVC
        probList = model.predict_proba([feat])[0]
        maxprobind = np.argmax(probList)
        prob = probList[maxprobind]
        outclass = model.classes_[maxprobind]
        
        
        result = {'img_url': img_url, 
                'prediction': outclass, 
                'probability': prob
                }

        return jsonify(result)

    except Exception as e:
        print(e)

@app.route('/predict_image', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)

        # Get the file from post request
        f = request.files['file']

        # load trained model
        model = joblib.load('example_model/vehicle-classification.p')

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # resize to 224x224x3
        img = cv.image(file_path)

        # extract features
        feat = img.getresnet50()
        probList = model.predict_proba([feat])[0]
        maxprobind = np.argmax(probList)
        prob = probList[maxprobind]
        
        outclass = model.classes_[maxprobind]
        
        return outclass

    return None

# model classifier builder
@app.route('/model_builder', methods=['GET', 'POST'])
def c_predict():
    message = ''    
    if request.method == 'POST':
        keywords = request.form.getlist('field[]')
        print(keywords)
        # input keywords to function and return a mod
        _, modFile = run(keywords)

        # model name
        model_name = modFile

        return render_template('download.html', value=model_name)
        # message = "Succesfully Register"
    return render_template('custom_predict.html', message=message)


# download file
@app.route('/return-files/<filename>')
def return_files(filename):
    # path to model
    path = os.path.join(current_app.root_path, 'classifier_model')
    
    return send_from_directory(directory=path, filename=filename, as_attachment=True)


if __name__=='__main__':
    app.run(host='127.0.0.1', port=5002)