
def get_disease(model, path):
  prediction = model.predict(path)
  return prediction[0], str(round(max(prediction[-1].tolist()) *100, 2)) + "%"