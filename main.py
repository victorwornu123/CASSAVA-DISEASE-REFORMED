# # ✅ PATCH MUST COME FIRST — before ANY fastai imports
# from fastai.learner import load_learner
# from fastai.learner import Learner
# import pathlib
# import platform
# import builtins

# # NOW import fastai and everything else
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import HTMLResponse
# # from fastapi.templating import Jinja2Templates
# from fastai.vision.all import *
# import joblib
# import uvicorn
# import shutil
# import os
# import uuid
# import torch
# from layer1 import leaf_vs_non_leaf
# from layer2 import get_anomaly_outliers
# from layer3 import get_disease

# import os
# os.environ['FASTAI_HOME'] = '/app'
# os.environ['TORCH_HOME'] = '/app'

# app = FastAPI(title="Cassava Leaf Disease Prediction API")

# # Lazy-load models on first request instead of at startup
# _models = {}

# def get_models():
#     global _models
#     if not _models:
#         _models['model_layer1'] = load_learner("models/model_layer_v1.pkl", cpu=True)
#         _models['model_layer3'] = load_learner("models/cassava_model_v1.pkl", cpu=True)
#         _models['bouncer'] = joblib.load("models/leaf_bouncer.pkl")
#         _models['scaler'] = joblib.load("models/leaf_scaler.pkl")
#     return _models

# # templates = Jinja2Templates(directory="templates")

# def workflow(path):
#   models = get_models()
#   if leaf_vs_non_leaf(models['model_layer1'], path) == "FAIL":
#     return "Enter a Valid Cassava Leaf Image!"
#   else:
#     if get_anomaly_outliers(path, models['model_layer3'], models['bouncer'], models['scaler']) == "FAIL":
#       return "Enter a Valid Cassava Leaf Image!"
#     else:
#       return get_disease(path)

# # print(workflow("C:/Users/Victor Wornu/Documents/fast ai learning/venv/code/train_images/1001742395.jpg"))
# # print("APP WORKS!!!")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
    
#     contents = await file.read()

#     # Run your model here
#     prediction = workflow(file)

#     return {
#         "prediction": prediction[0],
#         "confidence": prediction[1]
#     }


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# ✅ PATCH MUST COME FIRST — before ANY fastai imports
from fastai.learner import load_learner
from fastai.learner import Learner
import pathlib
import platform
import builtins

# NOW import fastai and everything else
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
from fastai.vision.all import *
import joblib
import uvicorn
import shutil
import os
import uuid
import torch
from layer1 import leaf_vs_non_leaf
from layer2 import get_anomaly_outliers
from layer3 import get_disease

import os
os.environ['FASTAI_HOME'] = '/app'
os.environ['TORCH_HOME'] = '/app'



app = FastAPI(title="Cassava Leaf Disease Prediction API")

test_image = "/app/test_images/leaf images/cmd2.jpg"

# Lazy-load models on first request instead of at startup
_models = {}

def get_models():
    global _models
    if not _models:
        _models['model_layer1'] = load_learner("models/model_layer_v1.pkl", cpu=True)
        _models['model_layer3'] = load_learner("models/cassava_model_v1.pkl", cpu=True)
        _models['bouncer'] = joblib.load("models/leaf_bouncer.pkl")
        _models['scaler'] = joblib.load("models/leaf_scaler.pkl")
    return _models

# templates = Jinja2Templates(directory="templates")

def workflow(path):
    models = get_models()
    model1 = models['model_layer1']
    model3 = models['model_layer3']
    bouncer = models['bouncer']
    scaler = models['scaler']
    if leaf_vs_non_leaf(model1, path) == "FAIL":
        return "Enter a Valid Cassava Leaf Image!", ""
    else:
        if get_anomaly_outliers(path, model3, bouncer, scaler) == "FAIL":
            return "Enter a Valid Cassava Leaf Image!", ""
        else:
            return get_disease(model3, path)

# print(workflow("C:/Users/Victor Wornu/Documents/fast ai learning/venv/code/train_images/1001742395.jpg"))
# print("APP WORKS!!!")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cassava-leaf-detector-reformed.netlify.app",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()

    # Run your model here
    temp_path = f"/tmp/{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
      f.write(contents)
    
    prediction = workflow(temp_path)
    print(prediction)
    return {
        "prediction": prediction[0],
        "confidence": prediction[1]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
