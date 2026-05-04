import torch
from fastai.vision.all import *

def get_embeddings(path, model):
    """Passes an image through the ResNet body to get a feature vector."""
    img = PILImage.create(path)

    # 1. Prepare image using fastai's internal pipeline
    dl = model.dls.test_dl([img])
    batch = dl.one_batch()

    # 2. Extract features from the 'body' (everything before the head)
    model_body = model.model[0].eval()

    with torch.no_grad():
        features = model_body(batch[0])
        # Flatten the (Batch, Channel, H, W) to (Batch, 2048)
        flat_features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)

    return flat_features.cpu().numpy()


def get_anomaly_outliers(img_path, learner, bouncer, scaler):
    """
    Final pipeline: Uses an SVM bouncer to filter anomalies
    before passing the image to the fastai disease classifier.
    """

    # --- LAYER 1: ANOMALY CHECK (The Bouncer) ---
    # 1. Get the 2048-dimensional fingerprint from fastai
    current_emb = get_embeddings(img_path, learner)

    # 2. Scale the fingerprint so the SVM can read it correctly
    scaled_emb = scaler.transform(current_emb)

    # 3. Predict: 1 means 'Inlier' (Leaf), -1 means 'Outlier' (Anomaly)
    is_leaf = bouncer.predict(scaled_emb)[0]

    if is_leaf == -1:
        return "FAIL"
    else:
        return "PASS"