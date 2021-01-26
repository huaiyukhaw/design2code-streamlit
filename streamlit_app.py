import streamlit as st
import base64
import io
import requests
import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import random

# use file uploader object to recieve image
# Remember that this bytes object can be used only once


def bytesioObj_to_base64str(bytesObj):
    return base64.b64encode(bytesObj.read()).decode("utf-8")

# Image conversion functions


def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


def PILImage_to_cv2(img):
    return np.asarray(img)


def ImgURL_to_base64str(url):
    return base64.b64encode(requests.get(url).content).decode("utf-8")


def drawboundingbox(img, boxes, pred_cls, scores, rect_th=10, text_size=3, text_th=10):
    img = PILImage_to_cv2(img)
    (H, W) = img.shape[:2]
    rect_th = round(int(W * 0.0025))
    text_size = round(int(W * 0.001))
    text_th = round(int(W * 0.0025))
    padding = round(int(W * 0.012))
    class_color_dict = {}
    for i in range(len(boxes)):
        
        class_color_dict[i] = [random.randint(0, 255) for _ in range(3)]
        
        left = int(boxes[i][0] * W)
        
        top = int(boxes[i][1] * H)
        
        width = int(boxes[i][2] * W)
        
        height = int(boxes[i][3] * H)
        
        start_point = (left, top)
        
        end_point = (left + width, top + height)
        
        cv2.rectangle(img, start_point, end_point, color=class_color_dict[i], thickness=rect_th)
                        
        cv2.putText(img, pred_cls[i] + " " + str(scores[i]), (left, top-padding), cv2.FONT_HERSHEY_SIMPLEX,
					text_size, class_color_dict[i], thickness=text_th)
                    
        plt.figure(figsize=(20, 30))
        
    plt.imshow(img)
    
    plt.xticks([])
    
    plt.yticks([])
    
    plt.show()
	
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("<h1>UI Detector with YOlOv4-Tiny</h1><br>", unsafe_allow_html=True)
st.markdown("<p>Built by <a href='https://twitter.com/huaiyukhaw' target='_blank'>@huaiyukhaw</a></p><br>", unsafe_allow_html=True)

bytesObj = st.file_uploader("Choose an image file")

st.markdown("<center><h2>or</h2></center>", unsafe_allow_html=True)

url = st.text_input('Enter URL')

if bytesObj or url:
    # In streamlit we will get a bytesIO object from the file_uploader
    # and we convert it to base64str for our FastAPI
    if bytesObj:
        base64str = bytesioObj_to_base64str(bytesObj)

    elif url:
        base64str = ImgURL_to_base64str(url)

    # We will also create the image in PIL Image format using this base64 str
    # Will use this image to show in matplotlib in streamlit
    img = base64str_to_PILImage(base64str)

    # Run FastAPI
    payload = json.dumps({
        "base64": base64str,
        "threshold": 0.4
    })

    response = requests.post("http://ui.huaiyukhaw.com/detect", data=payload)
    data = response.json()
    boxes = [d['box'] for d in data]
    pred_cls = [d['detectionString'] for d in data]
    scores = [round(d['score'], 4) for d in data]

    st.markdown("<center><h1>Result</h1></center>", unsafe_allow_html=True)
    drawboundingbox(img, boxes, pred_cls, scores)
    st.pyplot()
    st.markdown("<center><h1>FastAPI Response</h1></center><br>",
                unsafe_allow_html=True)
    st.write(data)
