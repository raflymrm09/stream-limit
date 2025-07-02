# app.py
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["red" if label=="person" else "green" for label in prediction["labels"]],
        width=2
    )
    return img_with_bboxes.detach().numpy().transpose(1, 2, 0)

st.title("Object Detection Web App")
upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload:
    img = Image.open(upload)
    prediction = make_prediction(img)
    img_np = np.array(img).transpose(2, 0, 1)
    img_with_bbox = create_image_with_bboxes(img_np, prediction)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_with_bbox)
    plt.axis('off')
    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    st.subheader("Detected Objects")
    st.write(prediction)
