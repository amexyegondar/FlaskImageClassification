from flask import Flask,request,jsonify
import werkzeug
import werkzeug.utils
from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

app=Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    if(request.method=='POST'):
        imageFile=request.files['image']
        filename=werkzeug.utils.secure_filename(imageFile.filename)
        imageFile.save("./uploadedImage/" +filename)
        return jsonify({
            "message":"Image uploaded successfully"
        })

@app.route("/predict")
def predict():
    new_model = load_model('model/imageclassifier.h5')
    img=cv2.imread('uploadedImage/happy.png')
    resize=tf.image.resize(img,(256,256))
    y_pred=new_model.predict(np.expand_dims(resize/255, 0))
    if (y_pred>0.5):
        print('she is sad')
    else:
        print('she is happy')

    

    


if __name__=="__main__":
    app.run(debug=True,port=4000)