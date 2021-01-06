import os
from PIL import Image

from flask import render_template, request
from flask import redirect, url_for

from app.utils import gender_prediction

UPLOAD_FOLDER = 'static/uploads'

def base():
    return render_template('base.html')

def index():
    return render_template('index.html')

def getWidth(path):

    img = Image.open(path)

    size = img.size
    aspect = size[0] / size[1]
    width = 300 * aspect

    return int(width)

def faceapp():

    if request.method == 'POST':

        f = request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        width = getWidth(path)

        # predictions

        gender_prediction(path, filename, color='bgr')

        return render_template('faceapp.html', fileupload=True, width=width, img_name=filename)

    return render_template('faceapp.html', fileupload=False, width="300", img_name="None")
