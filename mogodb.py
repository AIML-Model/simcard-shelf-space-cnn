import pymongo
from joblib import load
from flask import Flask, request, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
import logging
import pandas as pd
import os
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np
import openai
import cv2
import re
from datetime import datetime


def load_model(image_path):
    # Step 1: Load the trained YOLO model
    model = YOLO("/Users/ganesh/Desktop/Git-kyndryl/CAPSTONE-PROJECT/capstone-sim-shelf-space/simcard-shelf-space-2/runs/detect/train4/weights/best.pt")

    image = cv2.imread(image_path)

    # Run inference on the image
    results = model.predict(image, conf=0.25, iou=0.45, imgsz=800)
    return model, results


def get_image_path():
    directory = '/Users/ganesh/Desktop/Git-kyndryl/CAPSTONE-PROJECT/Retail-store-name'
    #    List all files and directories in the specified directory
    filenames_list = []
    dir_list = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
                filenames_list.append(file_path)
                folder = file_path.split('/')
                dir_list.append(folder[7])
    return filenames_list,dir_list


def mongo_connection():
    # Connect to the MongoDB cluster
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    return client

def insert_data(collection_name, data):
    client = mongo_connection()
    # Insert data into the MongoDB collection
    db = client["AIML"]
    collection = db[collection_name]
    collection.insert_one(data)
\
def all_store_image_process_pred():
    model = YOLO("/Users/ganesh/Desktop/Git-kyndryl/CAPSTONE-PROJECT/capstone-sim-shelf-space/simcard-shelf-space-2/runs/detect/train4/weights/best.pt")
    file_path,dir = get_image_path()
    document = {}
    lenght = len(file_path)
    for i in range(lenght):
        if i is not None:
            image_path = file_path[i]
            dir_name = dir[i]
            #print(image_path)
            image = cv2.imread(image_path)
            #print(image)
            results = model.predict(image, conf=0.25, iou=0.45, imgsz=800)
            for result in results[0].boxes:
                class_id = int(result.cls)
                class_name = model.names[class_id]
                document = {'Store_Name': dir_name.replace('_',' ')+' Philippines', 'Product_Name': class_name, "Insert_Date": datetime.now()}
                collection_name = "all_store_product_details"
                insert_data(collection_name, document)


def all_store_image_process_pred():
    model = YOLO("/Users/ganesh/Desktop/Git-kyndryl/CAPSTONE-PROJECT/capstone-sim-shelf-space/simcard-shelf-space-2/runs/detect/train4/weights/best.pt")
    file_path,dir = get_image_path()
    document = {}
    lenght = len(file_path)
    for i in range(lenght):
        if i is not None:
            image_path = file_path[i]
            dir_name = dir[i]
            #print(image_path)
            image = cv2.imread(image_path)
            #print(image)
            results = model.predict(image, conf=0.25, iou=0.45, imgsz=800)
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = model.names[class_id]
                class_id = int(results.cls)

                
                
                document = {'Store_Name': dir_name.replace('_',' ')+' Indonesia', 'Product_Name': class_name, "Insert_Date": datetime.now()}
                collection_name = "all_store_product_details"
                insert_data(collection_name, document)


all_store_image_process_pred()                    

# def process_image():
#     #model,result = load_model(image_path)    
#     file_path = get_image_path()
#     detected_objects = {}
#     lenght = len(file_path)
#     for i in range(lenght):
#         if i is not None:
#             image_path = file_path[i]
#             #print(image_path)
#             image = cv2.imread(image_path)
#             #print(image)
#             model, results = load_model(image_path)
#             for result in results[0].boxes:
#                 class_id = int(result.cls)
#                 class_name = model.names[class_id]
#                 if class_name in detected_objects:
#                     detected_objects[class_name] += 1
#                 else:
#                     detected_objects[class_name] = 1
#             sorted_dict = dict(sorted(detected_objects.items(), key=lambda item: item[1], reverse=True))
#             collection_name = "product_visibility_count"
#             insert_data(collection_name, sorted_dict)


# #process_image()

# # #print("====================================")
# # #print(results)
# # #print("====================================")

# def get_data(collection_name):
#     client = mongo_connection()
#     db = client["AIML"]
#     collection = db[collection_name]
#     pipeline = [
#     {
#         "$group": {
#             "_id": {
#                 "Store_Name": "$Store_Name",
#                 "Product_Name_Category": {
#                     "$cond": {
#                         "if": {
#                             "$or": [
#                                 { "$eq": ["$Product_Name", "im3"] },
#                                 { "$eq": ["$Product_Name", "Indosat-3"] },
#                                 { "$eq": ["$Product_Name", "Indosat"] }
#                             ]
#                         },
#                         "then": "Indosat",
#                         "else": "$Product_Name"
#                     }
#                 }
#             },
#             "product_count": { "$sum": 1 }
#         }
#     },
#     {
#         "$sort": { "_id.Store_Name": 1, "product_count": -1 }
#     }
# ]

#     result = collection.aggregate(pipeline)
#     return result

# Output= get_data("Tanjay")
# for data in Output:
#     store_name = data['_id']['Store_Name']
#     product_name_category = data['_id']['Product_Name_Category']
#     product_count = data['product_count']
#     print(store_name, product_name_category, product_count)
#     #print(data)
#     #print(data['Store_Name'], data['Product_Name_Category'], data['product_count'])