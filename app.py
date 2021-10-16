import numpy as np
from flask import Flask, request, Response, jsonify
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import json
import cv2
from flask_cors import CORS, cross_origin
import urllib

app = Flask(__name__)

CORS(app, support_credentials=True)

resnet_model = './models/ResNet50_Model.hdf5'
resnet_json = './models/ResNet50.json'

IMAGE_SIZE = 64

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)



@app.route('/predict',methods=["POST"])
@cross_origin()
def predict():
    # print(request.json)
    # convert string of image data to uint8
    # print(request.json['img'])
    # nparr = np.fromstring(request.json['img'], dtype="uint8")
    # print(nparr)
    # print(len(nparr))


    url = request.json['img']
    print(url)
    resp = urllib.request.urlopen(url)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    print(img.shape)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(img.shape)
    # decode image
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img)
    pred_arr = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    print(pred_arr)
    if img is not None:
            pred_arr[0] = resize_image(img, (IMAGE_SIZE, IMAGE_SIZE))
            
    pred_arr = pred_arr/255
    # print(pred_arr)    
    with open(resnet_json, 'r') as resnetjson:
            resnetmodel = model_from_json(resnetjson.read())
    resnetmodel.load_weights(resnet_model)
    
    label_resnet = resnetmodel.predict(pred_arr)
    print(label_resnet)
    idx_resnet = np.argmax(label_resnet[0])
    cf_score_resnet = np.amax(label_resnet[0])
    
    js_response= [
        {
            "idx_resnet":idx_resnet,
            "cf_score_resnet":cf_score_resnet
        }
    ]
    # print(js_response)
    return json.dumps(str(js_response))
    
    
@app.route("/test_api", methods=["POST"])
@cross_origin()
def test_api():
    print("OK")
    incoming_data = request.json
    print(incoming_data['param1'])
    return incoming_data
    

@app.route("/health_check", methods=["GET"])
@cross_origin()
def health_check():
    return "API Up And Running"

# Default port:
if __name__ == "__main__":
    app.run()