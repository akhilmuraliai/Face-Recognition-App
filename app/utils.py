import numpy as np 

from PIL import Image

import pickle
import cv2

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

harr = cv2.CascadeClassifier('./IPYNB-Notebooks/harrcascade-classifier/haarcascade_frontalface_default.xml')
mean = pickle.load(open('./IPYNB-Notebooks/models/XMean.pickle', 'rb'))
model_svm = pickle.load(open('./IPYNB-Notebooks/models/SVC_Model.pickle', 'rb'))
model_pca = pickle.load(open('./IPYNB-Notebooks/models/PCA_50.pickle', 'rb'))

gender = ['M', 'F']


def gender_prediction(path, filename, color='bgr'):

    img = cv2.imread(path)

    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = harr.detectMultiScale(gray, 1.5, 3)

    for x, y, w, h in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi = gray[y:y+h, x:x+w]
        roi = roi / 255

        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)

        roi_reshape = roi_resize.reshape(1, -1)

        roi_mean = roi_reshape - mean

        eigen_image = model_pca.transform(roi_mean)

        results = model_svm.predict_proba(eigen_image)[0]

        score = results[results.argmax()]

        score = score * 100.0

        text = "%s [%.0f" % (gender[results.argmax()], score) + "%]"

        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite('./static/predicted/{}'.format(filename), img)
