import streamlit as st
from PIL import Image
import pytesseract
from pytesseract import Output
import os
from pdf2image import convert_from_path
import cv2
import torch
import tempfile
import numpy as np
import layoutparser as lp
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)


#@st.cache(allow_output_mutation=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA



# Set the device to CPU
device = torch.device('cpu')


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.linkpicture.com/q/DP_for_Site.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



def perform_ocr(image):
    # Perform OCR using Tesseract
    image = np.array(image)


    extracted_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    # Get bounding box coordinates and text
    bounding_boxes = extracted_data["level"]
    texts = extracted_data["text"]
    
    # Visualize the text with bounding boxes
    image_with_boxes = image.copy()
    for i in range(len(bounding_boxes)):
        if bounding_boxes[i] == 5:
            (x, y, w, h) = (
                extracted_data["left"][i],
                extracted_data["top"][i],
                extracted_data["width"][i],
                extracted_data["height"][i]
            )
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, texts[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image_with_boxes


def detect_layout(image):

    image1 = image[..., ::-1]
    
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    
    layout = model.detect(image1)
    color = (0, 255, 0)  # Green color (BGR format)
    thickness = 2  # Line thickness

    for i in range(len(layout)):
        x1 = int(layout._blocks[i].block.x_1)
        x2 = int(layout._blocks[i].block.x_2)
        y1 = int(layout._blocks[i].block.y_1)
        y2 = int(layout._blocks[i].block.y_2)
        image_1 = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
    color_converted = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_converted)
    
    return pil_image

    


def main():
    add_bg_from_url() 
    st.title("OCR with Bounding Box Visualization")
    st.write("Upload a PNG or PDF document to perform OCR and visualize the results.")
    
    uploaded_file = st.file_uploader("Upload Document", type=["png", "pdf"])
    
    st.markdown(
        """
        <style>
        body {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        if file_extension == ".png":
            # Read and display the image
            print(uploaded_file.name)
            
            file_name = uploaded_file.name
            image = Image.open(uploaded_file)
            image_cv = Image.open(uploaded_file).convert('RGB') 
            image_cv = np.array(image_cv)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            ocr_result = perform_ocr(image)
            original_title = '<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>OCR Results Bounding Boxes</b></p>'
            st.markdown(original_title, unsafe_allow_html=True)
    
            st.image(ocr_result,use_column_width=True)
            
            # Apply layout analysis or table detection
            layout_result = detect_layout(image_cv)
            original_title = '<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>Layout Results Bounding Boxes</b></p>'
            st.markdown(original_title, unsafe_allow_html=True)
            st.image(layout_result,use_column_width=True)
        
        elif file_extension == ".pdf":
            # Convert PDF to images
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_file.read())
                
            images = convert_from_path(temp_path)
            
            for i, image in enumerate(images):
                st.image(image, caption=f"Page {i+1}", use_column_width=True)
                
                # Perform OCR and visualize with bounding boxes
                ocr_result = perform_ocr(image)
                original_title = '<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>OCR Results Bounding Boxes</b></p>'
                st.markdown(original_title, unsafe_allow_html=True)
                st.image(ocr_result, use_column_width=True)
                
                # # Apply layout analysis or table detection
                image_cv = image.convert('RGB') 
                image_cv = np.array(image_cv)
                layout_result = detect_layout(image_cv)
                original_title = '<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>Layout Results Bounding Boxes</b></p>'
                st.markdown(original_title, unsafe_allow_html=True)
                st.image(layout_result, use_column_width=True)
                # st.write(layout_result)


if __name__ == "__main__":
    main()
