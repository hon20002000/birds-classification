import time
from flask import Flask, render_template, flash, redirect, request, url_for
from flask_sqlalchemy import SQLAlchemy
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize
import json
from PIL import Image
import psycopg2 #pip install psycopg2 
import psycopg2.extras
import urllib.request
import os
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as T
import numpy as np
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
from torchvision import transforms


app = Flask(__name__)
     
DB_HOST = "postgres"
DB_NAME = "students"
DB_USER = "postgres"
DB_PASS = "1234"
     
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=5432)

UPLOAD_FOLDER = 'static/uploads/'
  
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def predict(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('./model/id2label.json', 'r') as f:
        id2label = json.load(f)
    with open('./model/label2id.json', 'r') as f:
        label2id = json.load(f)
    
    feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-xlarge-224-22k")
    model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224",
                                                        num_labels=len(id2label),
                                                        id2label=id2label,
                                                        label2id=label2id,
                                                        ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load('./model/tiny_model_221110_epoch_2.pt'))
    model.to(device).eval()
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    transform = Compose([Resize((224, 224)),
                 ToTensor(),
                 normalize])
    img = Image.open(img_path)
    img = transform(img.convert("RGB"))
    img = img.unsqueeze(dim=0)
    outputs = model(img)['logits'].argmax(-1).item()
    y_pred = id2label[str(outputs)]
    return y_pred


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cursor.execute("CREATE TABLE IF NOT EXISTS upload (title VARCHAR (250), prediction VARCHAR (100))")
        conn.commit()
        flash('Image successfully uploaded !')
        img_path = 'static/uploads/' + filename
        y_pred = predict(img_path)
        cursor.execute("INSERT INTO upload (title, prediction) VALUES (%s, %s)", (filename, y_pred))  #execute insert sql
        conn.commit()
        return render_template('index.html', filename=filename, y_pred=y_pred)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0')  #host
