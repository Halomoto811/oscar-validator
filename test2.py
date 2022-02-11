import datetime
from datetime import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO


SUB_KEY = "ad304aad25f74e61b072c35447d20642"
ENDPOINT_URL = "https://azure-face-test-1.cognitiveservices.azure.com/"
assert SUB_KEY
# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT_URL, CognitiveServicesCredentials(SUB_KEY))

# 30 days
today=datetime.datetime.now()
period=today+datetime.timedelta(days=-30)

# use SDK
def get_face_api(img):
    #with io.BytesIO() as temp:
    temp = io.BytesIO()
    img.save(temp, format="JPEG")
    temp.seek(0)

    requireAttributes = ['age','gender','headPose','smile','facialHair','glasses','emotion','hair','makeup','occlusion','accessories','blur','exposure','noise']
    detected_faces = face_client.face.detect_with_stream(temp,return_face_id=True,detection_model='detection_01',return_face_attributes=requireAttributes)
    if not detected_faces:
        return []
    # for each face define position of rectangle
    #arial = ImageFont.truetype("/Library/Fonts/NewYork.ttf", 40)
    for index,face in enumerate(detected_faces):
        rect = face.face_rectangle
        left = rect.left
        top = rect.top
        width = rect.width
        height = rect.height
        draw = ImageDraw.Draw(img)
        draw.rectangle([(left,top),(left+width,top+height)],fill=None,outline='red',width=5)
        #draw.text((left+20,top+20),str(index),fill='red',font=arial)
    return [img,detected_faces]

# compare faces
def compare_faces(img2,faces):
    img = img2

# return multiple lists
def handle_result_dataframe(faces):
    age_list = []
    gender_list = []
    smile_list = []
    haircolor_list = []
    for result in faces:
        faceAttributes = result.face_attributes

        age_list.append(int(faceAttributes.age))
        gender_list.append(faceAttributes.gender)
        # handle smile
        if faceAttributes.smile == 1:
            smile_list.append('Yes')
        elif faceAttributes.smile == 0:
            smile_list.append('No')
        else:
            smile_list.append('Maybe?')
        # get the most possible hair color
        if len(faceAttributes.hair.hair_color) > 0:
            final_color = "unknown"
            highest_point = 0.0
            for this_color in faceAttributes.hair.hair_color:
                if this_color.confidence > highest_point:
                    final_color = this_color.color
            haircolor_list.append(final_color)
        else:
            haircolor_list.append('unknown')

    return {
            "age": age_list,
            "gender": gender_list,
            "smile": smile_list,
            "hair_color": haircolor_list
            }
########### covid api ############
def get_reported_kens(all_json):
    ken_list = ["全国"]
    kens = all_json[len(all_json)-1]
    for ken in kens["area"]:
        ken_list.append(ken["name_jp"])
    return ken_list

def get_result_by_ken(all_json,ken):
    date_list_2 = []
    number_list_2 = []
    if ken == "全国":
        for daily in all_json[len(all_json)-15:len(all_json)-1]:
            date_list_2.append(daily["lastUpdate"])
            number_list_2.append([daily["npatients"],daily["ncurrentpatients"]])
    else:
        for daily in all_json[len(all_json)-15:len(all_json)-1]:
            for area in daily["area"]:
                if area["name_jp"] == selected_p:
                    date_list_2.append(daily["lastUpdate"])
                    number_list_2.append([area["npatients"],area["ncurrentpatients"]])
    return [date_list_2,number_list_2]

########### UI #############
st.sidebar.title("Oscar's Project")
main_radio = st.sidebar.radio("Choose a page",("Show Visualization","Show Uploader"))
#if st.sidebar.checkbox("Show Uploader"):
if main_radio == "Show Uploader":
    """
    @ Oscar's face validator test
    #### Validate your face!
    """
    selected = st.radio("Choose a method to upload",('Upload a file','Camera'))
    if selected == 'Upload a file':
        upload_file = st.file_uploader("Choose your photo",type=["jpg","jpeg"])
    else:
        upload_file = st.camera_input("Open your camera and.. SMILE! :) ")
    # if uploaded then detect and print
    if upload_file is not None:
        img = Image.open(upload_file)
        # get drawed image and al detected faces from api
        with st.spinner("Uploading..."):
            drawed_img_and_detected_faces=get_face_api(img)
            if len(drawed_img_and_detected_faces) == 0:
                st.warning("No face detected.")
                st.stop()
            detected_faces = drawed_img_and_detected_faces[1]
            st.success("Recoganized faces: " + str(len(detected_faces)))
            st.image(drawed_img_and_detected_faces[0],'Uploaded image',width=100,use_column_width=True)
            # handle spreadsheet
            result_dataframe = handle_result_dataframe(detected_faces)

            #if st.checkbox("Show spreadsheet"):
            with st.expander("Show more detection Result: ",expanded=False):
                st.caption("* Calculated by Microsoft, implemented by Oscar.")
                st.write(pd.DataFrame(result_dataframe))

            # for uploaded photos only, show option to compare faces
            """
            #### Compare with another photo
            """
            if st.checkbox("Compare with another photo"):
                if (len(detected_faces) > 1):
                    st.warning("More than 1 face detected. Please choose only one.")
                    options = []
                    for idx,option in enumerate(detected_faces):
                        option_age = result_dataframe['age'][idx]
                        option_gender = result_dataframe['gender'][idx]
                        options.append(str(idx) + " - Age: " + option_age + ", ")
                    selected_face = st.selectbox("Please select only one face.",options)
                selected_second = st.radio("Choose a method to upload again",('Upload a file','Camera'))
                if selected_second == 'Upload a file':
                    upload_second = st.file_uploader("Choose another photo",type=["jpg","jpeg","png"])
                else:
                    upload_second = st.camera_input("Open your camera and.. SMILE! :) ")
                if upload_second is not None:
                    # pass face.face_id to api and compare
                    img2 = Image.open(upload_second)
                    compared_face_id = compare_faces(img2,detected_faces)

elif main_radio == "Show Visualization":
#if st.sidebar.checkbox("Show Visualization"):
    """
    #### COVID-19 Data Visualization
    """
    with st.spinner("Processing..."):
        st.caption("* Data source: Japan Goverment (https://corona.go.jp/dashboard/)")
        all_json = requests.get("https://www.stopcovid19.jp/data/covid19japan-all.json").json()
        kens = get_reported_kens(all_json)

        selected_p = st.selectbox("Select a prefecture.",kens)
        ken_report = get_result_by_ken(all_json,selected_p)
        covid_df = pd.DataFrame(
            ken_report[1],
            columns=["合計","新規"],
            index=ken_report[0]
            )

        st.empty()
        last = ken_report[1][len(ken_report[1])-1][0]
        last2 = ken_report[1][len(ken_report[1])-2][0]
        st.metric(selected_p+" (Last Update: "+ken_report[0][len(ken_report[0])-1]+")",str(last)+" 件",str(last-last2)+" 件")
        with st.container():
            st.bar_chart(covid_df)

        #df = pd.DataFrame(
            #[[139.62,35.66],[139.2,35.22],[139.11,35.222],[139.11,35.225]],
            #lonlats,
            #columns=['lon', 'lat'])
        #st.map(df)

########################################################
# call azure face api and return as [img,respnse(json)]
def call_face_api(img):
    api_url = ENDPOINT_URL + "/face/v1.0/detect"
    with io.BytesIO() as temp:
        img.save(temp, format="JPEG")
        binary = temp.getvalue()

    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key':SUB_KEY
    }
    params = {
        'detectionModel': 'detection_01',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
        'returnFaceId': 'true'
    }
    res = requests.post(api_url,params=params,headers=headers,data=binary)
    results = res.json()

    # for each face define position of rectangle
    arial = ImageFont.truetype("NewYork.ttf", 40)
    for index,result in enumerate(results):
        rect = result['faceRectangle']
        top = rect['top']
        left = rect['left']
        width = rect['width']
        height = rect['height']
        # draw, from left top to right bottom
        draw = ImageDraw.Draw(img)
        draw.rectangle([(left,top),(left+width,top+height)],fill=None,outline='red',width=5)
        draw.text((left+20,top+20),str(index),fill='red',font=arial)
    return [img,results]
