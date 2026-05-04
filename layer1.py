from fastai.vision.all import *



def leaf_vs_non_leaf(model, path):
  if model.predict(path)[0] == "leaf":
    return "PASS"
  else: 
    return "FAIL"