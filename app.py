import numpy as np
from flask import Flask, request, Response, jsonify
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import json
import cv2
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

resnet_model = './models/ResNet50_Model.hdf5'
resnet_json = './models/ResNet50.json'

IMAGE_SIZE = 64

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)



@app.route('/predict',methods=["POST"])
def predict():
    # convert string of image data to uint8
    nparr = np.fromstring(request.json['img'], np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    pred_arr = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    
    if img is not None:
            pred_arr[0] = resize_image(img, (IMAGE_SIZE, IMAGE_SIZE))
            
    pred_arr = pred_arr/255
            
    with open(resnet_json, 'r') as resnetjson:
            resnetmodel = model_from_json(resnetjson.read())
    resnetmodel.load_weights(resnet_model)
    
    label_resnet = resnetmodel.predict(pred_arr)
    idx_resnet = np.argmax(label_resnet[0])
    cf_score_resnet = np.amax(label_resnet[0])
    
    js_response= [
        {
            "idx_resnet":idx_resnet,
            "cf_score_resnet":cf_score_resnet
        }
    ]
    print(js_response)
    return json.dumps(str(js_response))
    
    
@app.route("/test_api", methods=["POST"])
def test_api():
    print("OK")
    incoming_data = request.json
    print(incoming_data['param1'])
    return incoming_data
    

@app.route("/health_check", methods=["GET"])
def health_check():
    return "API Up And Running"

# Default port:
if __name__ == "__main__":
    app.run()