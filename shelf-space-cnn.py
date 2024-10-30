from joblib import load
from flask import Flask, request, redirect, render_template, flash, url_for, jsonify
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
import pymongo
from datetime import datetime

new_path = '/Users/ganesh/Desktop/Git-kyndryl/CAPSTONE-PROJECT/capstone-sim-shelf-space'  # Replace with the desired path
os.chdir(new_path)

logging.basicConfig(level=logging.DEBUG,  # Set the logging level to DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from flask import Flask, request, jsonify
app = Flask(__name__)

# Specify the folder to save uploaded images
UPLOAD_FOLDER = 'static/original'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'  # Required for flashing messages

# Check if upload folder exists; if not, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def common():
    # Step 1: Load the trained YOLO model
    model = YOLO("simcard-shelf-space-2/runs/detect/train4/weights/best.pt")
    
    file2 = open('image_file.conf', 'r')
    image_name = file2.read()
    file2.close()
    image_path = "static/original/" +image_name  # Replace with your test image path
    # Load and prepare the image
    image = cv2.imread(image_path)

    # Run inference on the image
    results = model.predict(image, conf=0.25, iou=0.45, imgsz=800)
    for result in results:
        result.save("static/predicted/output_image_with_detections.jpg") 
    return model, results, image_name, image

def extract_text_from_image(image_path):
    # Open image
    img = cv2.imread(image_path)
    custom_config=r'-l eng --oem 3 --psm 6'
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(img, config=custom_config)    
    return text

def display_lines_matching_pattern(text, pattern):
    # Split the text into lines
    lines = text.splitlines()

    # Iterate over each line
    for line in lines:
        # Check if the pattern matches the line
        if re.search(pattern, line):
            #output_text = re.sub(r'Name&Address:', '', line)
            output_text = re.sub(r'[—   -]', '', line)
            output = line.lstrip(',').lstrip('-').rstrip('-')
    return output

def image_to_text():
    model, results, image_name, image = common()
    # Extract text from the image
    image_path = "static/original/" +image_name
    pattern = r'Address'
    extracted_text = extract_text_from_image(image_path)
    text = display_lines_matching_pattern(extracted_text, pattern)
    # Display the extracted text
    return text

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

def get_data(collection_name, Store_Name):
    client = mongo_connection()
    db = client["AIML"]
    collection = db[collection_name]
    pipeline = [
        {
        "$match": {
            "Store_Name":  Store_Name
        }
    },
    {
        "$group": {
            "_id": {
                "Store_Name": "$Store_Name",
                "Product_Name_Category": {
                    "$cond": {
                        "if": {
                            "$or": [
                                { "$eq": ["$Product_Name", "im3"] },
                                { "$eq": ["$Product_Name", "Indosat-3"] },
                                { "$eq": ["$Product_Name", "Indosat"] }
                            ]
                        },
                        "then": "Indosat",
                        "else": "$Product_Name"
                    }
                }
            },
            "product_count": { "$sum": 1 }
        }
    },
    {
        "$sort": { "_id.Store_Name": 1, "product_count": -1 }
    }
]

    result = collection.aggregate(pipeline)
    return result

def get_data_all_store(collection_name):
    client = mongo_connection()
    db = client["AIML"]
    collection = db[collection_name]
    pipeline = [
    {
        "$group": {
            "_id": {
                "Store_Name": "$Store_Name",
                "Product_Name_Category": {
                    "$cond": {
                        "if": {
                            "$or": [
                                { "$eq": ["$Product_Name", "im3"] },
                                { "$eq": ["$Product_Name", "Indosat-3"] },
                                { "$eq": ["$Product_Name", "Indosat"] }
                            ]
                        },
                        "then": "Indosat",
                        "else": "$Product_Name"
                    }
                }
            },
            "product_count": { "$sum": 1 }
        }
    },
    {
        "$sort": { "_id.Store_Name": 1, "product_count": -1 }
    }
]

    result = collection.aggregate(pipeline)
    return result

def get_data_all_store_count_graph(collection_name):
    client = mongo_connection()
    db = client["AIML"]
    collection = db[collection_name]
    pipeline = [
        {
            "$project": {
                "Store_Name": 1,
                "Product_Group": {
                    "$cond": {
                        "if": {
                            "$in": ["$Product_Name", ["im3", "Indosat-3", "Indosat"]]
                        },
                        "then": "Indosat",
                        "else": "others"
                    }
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "Store_Name": "$Store_Name",
                    "Product_Group": "$Product_Group"
                },
                "product_count": { "$sum": 1 }
            }
        },
        {
            "$group": {
                "_id": "$_id.Store_Name",
                "Indosat": {
                    "$sum": {
                        "$cond": [
                            { "$eq": ["$_id.Product_Group", "Indosat"] }, "$product_count", 0
                        ]
                    }
                },
                "Others": {
                    "$sum": {
                        "$cond": [
                            { "$eq": ["$_id.Product_Group", "others"] }, "$product_count", 0
                        ]
                    }
                }
            }
        },
        {
            "$sort": { "_id": 1 }  # Sort by Store_Name
        }
    ]
    result = collection.aggregate(pipeline)
    return result

def get_data_brand_compliance(collection_name, Store_Name):
    client = mongo_connection()
    db = client["AIML"]
    collection = db[collection_name]
    pipeline = [
    {
        "$match": {
            "Store_Name": Store_Name  # Filter for a specific Store_Name, modify as needed
        }
    },
    {
        "$project": {
            "_id": 1,  # Include the ObjectId
            "Store_Name": 1,  # Include Store_Name
            "Product_Name": 1,  # Include Product_Name
            "Position_X": 1,  # Include Position_X
            "Position_Y": 1,  # Include Position_Y
            "Height": 1,  # Include Height
            "Width": 1   # Include Width
        }
    }
]
    result = collection.aggregate(pipeline)
    return result

def get_data_brand_space(collection_name, Store_Name):
    client = mongo_connection()
    db = client["AIML"]
    collection = db[collection_name]
    pipeline = [
    {
        "$match": {
            "Store_Name": Store_Name
        }
    },
     {
        "$project": {
            "Store_Name": 1,
            "Product_Name": 1,
            "Total_Area": 1,
            "Product_Area": 1  # Include the Product_Area field
        }
    }
]
    
    result = collection.aggregate(pipeline)
    return result



def get_data_brand_space_graph(collection_name, Store_Name):
    client = mongo_connection()
    db = client["AIML"]
    collection = db[collection_name]
    pipeline = [
    {
        "$match": {
            "Store_Name": Store_Name  # Filter for a specific Store_Name, modify as needed
        }
    },
    {
        "$project": {
            "Store_Name": 1,
            # Sum of 'indosat', 'indosat-3', 'im3'
            "Indosat": {
                "$add": [
                    {"$ifNull": ["$indosat", 0]},
                    {"$ifNull": ["$indosat-3", 0]},
                    {"$ifNull": ["$im3", 0]}
                ]
            },
            # Calculate 'others' for fields not 'indosat', 'im3', 'indosat-3'
            "Compititor": {
                "$reduce": {
                    "input": {"$objectToArray": "$$ROOT"},
                    "initialValue": 0,
                    "in": {
                        "$cond": {
                            "if": {"$in": ["$$this.k", ["Indosat", "Im3", "Indosat-3", "Store_Name", "Total_Area", "_id"]]},
                            "then": "$$value",  # Skip these fields
                            "else": {"$add": ["$$value", "$$this.v"]}  # Sum all other fields
                        }
                    }
                }
            }
        }
    }
]

    result = collection.aggregate(pipeline)
    return result

def get_image_path():
    directory = '/Users/ganesh/Desktop/Git-kyndryl/CAPSTONE-PROJECT/Retail-store-name'
    #    List all files and directories in the specified directory
    filenames_list = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
                filenames_list.append(file_path)
    return filenames_list

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


@app.route('/home')
def home():
    return render_template('main.html')  # Assuming you have an index.html

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')  # Assuming you have an index.html

@app.route('/architecture')
def architecture():
    return render_template('architecture.html')  # Assuming you have an index.html

@app.route('/model')
def modeldetails():
    return render_template('model.html')  # Assuming you have an index.html

@app.route('/upload')
def uploadimage_init():
    return render_template('upload.html')  # Assuming you have an index.html

@app.route('/all-store-wise-data')
def all_store_wise_data_init():
    return render_template('all-store-wise-data.html')

@app.route('/predict')
def prediction_init():
    return render_template('predict.html')

@app.route('/retail_store')
def retail_store_init():
    return render_template('retail_store.html')

@app.route('/retail-store-count-graph')
def retail_store_count_init():
    return render_template('retail-store-count-graph.html')

@app.route('/iteam-count')
def iteam_wise_count_init():
    return render_template('iteam-wise-count.html')

@app.route('/iteam-space')
def iteam_space_init():
    return render_template('iteam-space.html')

@app.route('/iteam-wise-count-graph')
def iteam_wise_count_graph_init():
    return render_template('iteam-wise-count-graph.html')

@app.route('/iteam-space-graph')
def iteam_space_graph_init():
    return render_template('iteam-space-graph.html')

#@app.route('/location')
#def location_init():
#    return render_template('location.html')

@app.route('/stock-availability')
def stock_availability_init():
    return render_template('stock-availability.html')

@app.route('/brand-compliance')
def brand_compliance_init():
    return render_template('brand-compliance.html')



@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('uploadimage_init'))
    
    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('uploadimage_init'))

    # If the file is allowed, save it to the upload folder
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Secure the filename
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # Save the file
        flash('Image successfully uploaded')
        file1 = open('image_file.conf', 'w')
        file1.write(filename)
        file1.close()
        return redirect(url_for('uploadimage_init'))  # Redirect to the predict route

@app.route('/predict', methods=['POST'])
def predict():
    model, results, image_name, image = common()
    image_name = "original/" +image_name
    return render_template('predict.html', image_name=image_name)

@app.route('/iteam-wise-count' , methods=['POST'])
def iteam_wise_count():
    model, results, image_name, image = common()
    #store_name = image_to_text()
    store_name = ""
    detected_objects = {}
    indosat_group = ['Indosat-3', 'Indosat', 'im3']
    summed_value = 0

    for result in results[0].boxes:
        class_id = int(result.cls)  # Get the class ID of the object
        class_name = model.names[class_id]  # Convert class ID to class name
        # Update count for each detected object type
        if class_name in detected_objects:
            detected_objects[class_name] += 1
        else:
            detected_objects[class_name] = 1
    
    print(type(detected_objects))
    for key, value in detected_objects.items():
        if key in indosat_group:
            summed_value += value


    filtered_dict = {key: value for key, value in detected_objects.items() if key not in indosat_group}
    filtered_dict['Indosat'] = summed_value
    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
    #sorted_dict = dict(sorted(detected_objects.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
    return render_template('iteam-wise-count.html', sorted_dict=sorted_dict, image_name=image_name, store_name=store_name)

@app.route('/iteam-space',  methods=['POST'])
def iteam_space():
    model, results, image_name, image = common()
    # Get image height and width
    image_height, image_width = results[0].orig_shape[:2]
    #store_name = image_to_text()
    store_name = ""
    # Prepare a list to hold the formatted results
    output_data = []
    indosat_group = ['Indosat-3', 'Indosat', 'im3']
    product_area_sum = 0
    # Extract the information from the result
    for result in results:
         # Loop through each detected box
        for box in result.boxes:
            # Get the bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Get class ID and class name
            class_id = int(box.cls[0])  # Class id
            class_name = model.names[class_id]  # Class name from the model

            # Calculate bounding box width and height
            box_width = x2 - x1
            box_height = y2 - y1

            output_data.append({
                "class_name": class_name,
                "height": float(box_height),
                "width": float(box_width)
                })

    # Convert the result to JSON format
    json_output = json.dumps(output_data, indent=4)

    # Print the JSON output
    print(json_output)

    DPI = 96
    # 1 inch = 2.54 cm,  1 inch = 96 pixels, Therefore, 1 pixel = 2.54 cm / 96 ≈ 0.026458 cm
    # Optionally, save the JSON output to a file
    #Image total area
    height, width, _ = image.shape
    Total_image_area = round((height*width)*2.54/DPI)
    
    with open('yolov8_box_prediction.json', 'w') as f:
        f.write(json_output)
    df = pd.read_json('yolov8_box_prediction.json')
    df['Total_area'] = round((df['height']*df['width'])*2.54/DPI)
    df_final = df.groupby(['class_name'])['Total_area'].sum().reset_index()
    result_dict = df_final.set_index('class_name')['Total_area'].to_dict()
   
    for key, value in result_dict.items():
        if key in indosat_group:
            product_area_sum += value
    
    filtered_dict = {key: value for key, value in result_dict.items() if key not in indosat_group}
    filtered_dict['Indosat'] = product_area_sum
    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
    return render_template('iteam-space.html', result_dict=sorted_dict,Total_image_area=Total_image_area, image_name=image_name, store_name=store_name)



@app.route('/iteam-wise-count-graph',  methods=['POST'])
def iteam_wise_count_graph():
    document_data = []
    storename = request.form.get('dropdown')
    #storename = "all_store_product_details"
    collection_name = "all_store_product_details"
    print(storename)
    data= get_data(collection_name, storename)
    for data in data:
        print(data)
        store_name = data['_id']['Store_Name']
        product_name = data['_id']['Product_Name_Category']
        product_count = data['product_count']
        doc = {'Store_Name': store_name, 'Product_Name': product_name, 'Product_Count': product_count}
        document_data.append(doc)
        #document.append({'Store_Name': store_name, 'Product_Name': product_name, 'Product_Count': product_count})
        print(document_data)
    return render_template('iteam-wise-count-graph.html', sorted_dict=document_data)

@app.route('/iteam-space-graph', methods=['POST'])
def iteam_space_graph():
        model, results, image_name, image = common()
        #store_name = image_to_text()
        store_name = ""
        product_area_sum = 0
        indosat_group = ['Indosat-3', 'Indosat', 'im3']
        # Get image height and width
        image_height, image_width = results[0].orig_shape[:2]

        # Prepare a list to hold the formatted results
        output_data = []

    # Extract the information from the result
        for result in results:
             # Loop through each detected box
            for box in result.boxes:
                # Get the bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Get class ID and class name
                class_id = int(box.cls[0])  # Class id
                class_name = model.names[class_id]  # Class name from the model

                # Calculate bounding box width and height
                box_width = x2 - x1
                box_height = y2 - y1

                output_data.append({
                    "class_name": class_name,
                    "height": float(box_height),
                    "width": float(box_width)
                    })

        # Convert the result to JSON format
        json_output = json.dumps(output_data, indent=4)

        # Print the JSON output
        print(json_output)

        DPI = 96
        # 1 inch = 2.54 cm,  1 inch = 96 pixels, Therefore, 1 pixel = 2.54 cm / 96 ≈ 0.026458 cm
        # Optionally, save the JSON output to a file
        #Image total area
        height, width, _ = image.shape
        Total_image_area = round((height*width)*2.54/DPI)
    
        with open('yolov8_box_prediction.json', 'w') as f:
            f.write(json_output)
        df = pd.read_json('yolov8_box_prediction.json')
        # Total area in cm
        df['Total_area_cm'] = round((df['height']*df['width'])*2.54/DPI)
        # Total area sum in cm
        df_final = df.groupby(['class_name'])['Total_area_cm'].sum().reset_index()

        # Total area percentage
        df_final['Total_area_percentage'] = (df_final['Total_area_cm'])/Total_image_area*100
        # Total area percentage dictionary

        result_dict = df_final.set_index('class_name')['Total_area_percentage'].to_dict()
        # Total area cm dictionary
        
        result_dict1 = df_final.set_index('class_name')['Total_area_cm'].to_dict()
        # New dictionary with free space
        final_dict = {}
        final_dict['Empty Space'] = (Total_image_area - sum(result_dict1.values()))/Total_image_area*100
        # Update the final dictionary with the result dictionary percentage values
        final_dict.update(result_dict)
        #print(final_dict)
        #print(result_dict1)
        for key, value in final_dict.items():
            if key in indosat_group:
                product_area_sum += value
        filtered_dict = {key: value for key, value in final_dict.items() if key not in indosat_group}
        filtered_dict['Indosat'] = product_area_sum
        return render_template('iteam-space-graph.html', final_dict=filtered_dict, image_name=image_name, store_name=store_name)


@app.route('/stock-availability', methods=['POST'])
def stock_availability():
        model, results, image_name, image = common()
        #store_name = image_to_text()
        store_name = ""
        indosat_group = ['Indosat-3', 'Indosat', 'im3']
        summed_value = 0
        # Get image height and width
        image_height, image_width = results[0].orig_shape[:2]

        detected_objects = {}
        final_dict = {}

        for result in results[0].boxes:
            class_id = int(result.cls)  # Get the class ID of the object
            class_name = model.names[class_id]  # Convert class ID to class name
            # Update count for each detected object type
            if class_name in detected_objects:
                detected_objects[class_name] += 1
            else:
                detected_objects[class_name] = 1

        final_dict['Not Available'] = (100 - sum(detected_objects.values()))/100*100 
        for key, value in detected_objects.items():
                detected_objects[key] = ((value / 100) * 100)
        final_dict.update(detected_objects)   
        print(detected_objects)
        for key, value in final_dict.items():
            if key in indosat_group:
                summed_value += value
        filtered_dict = {key: value for key, value in detected_objects.items() if key not in indosat_group}
        filtered_dict['Indosat'] = summed_value
        return render_template('stock-availability.html', final_dict=filtered_dict, image_name=image_name, store_name=store_name)


@app.route('/brand-compliance',  methods=['POST'])
def brand_compliance():
    document_data = []
    storename = request.form.get('dropdown')
    #storename = "all_store_product_details"
    collection_name = "brand_compliance_details"
    #print(storename)
    data= get_data_brand_compliance(collection_name, storename)
    for data in data:
        print(data)
        store_name = data['Store_Name']
        product_name = data['Product_Name']
        Position_X = data['Position_X']
        Position_Y = data['Position_Y']
        Height = data['Height']
        Width = data['Width']
        doc = {'Store_Name': store_name, 'Product_Name': product_name, 'Position_X': Position_X, 'Position_Y': Position_Y, 'Height': Height, 'Width': Width}
        document_data.append(doc)
        #document.append({'Store_Name': store_name, 'Product_Name': product_name, 'Product_Count': product_count})
        print(document_data)
    print(document_data)
    return render_template('brand-compliance.html', result_dict=document_data)

@app.route('/retail-store',  methods=["POST"])
def retail_store():
    document_data = []
    #storename = request.form.get('dropdown')
    #storename = "all_store_product_details"
    collection_name = "all_store_product_details"
    #print(storename)
    data= get_data_all_store(collection_name)
    for data in data:
        print(data)
        store_name = data['_id']['Store_Name']
        product_name = data['_id']['Product_Name_Category']
        product_count = data['product_count']
        doc = {'Store_Name': store_name, 'Product_Name': product_name, 'Product_Count': product_count}
        document_data.append(doc)
        #document.append({'Store_Name': store_name, 'Product_Name': product_name, 'Product_Count': product_count})
        print(document_data)
    print(document_data)
    return render_template('retail_store.html', document=document_data)


@app.route('/retail-store-count-graph',  methods=['POST'])
def retail_store_count_graph():
    data_out = []
    #storename = request.form.get('dropdown')
    #storename = "all_store_product_details"
    collection_name = "all_store_product_details"
    #print(storename)
    result= get_data_all_store_count_graph(collection_name)
    for entry in result:
            data_out.append({
            "Store_Name": entry['_id'],
            "Indosat": entry['Indosat'],
            "Others": entry['Others']
            })
    return render_template('retail-store-count-graph.html', data=data_out)
    #print(detected_objects)

@app.route('/all-store-wise-data',  methods=["POST"])
def all_store_wise_data():
    document_data = []
    storename = request.form.get('dropdown')
    #storename = "all_store_product_details"
    collection_name = "all_store_product_details"
    print(storename)
    data= get_data(collection_name, storename)
    for data in data:
        print(data)
        store_name = data['_id']['Store_Name']
        product_name = data['_id']['Product_Name_Category']
        product_count = data['product_count']
        doc = {'Store_Name': store_name, 'Product_Name': product_name, 'Product_Count': product_count}
        document_data.append(doc)
        #document.append({'Store_Name': store_name, 'Product_Name': product_name, 'Product_Count': product_count})
        print(document_data)
    #document = {k: v for d in document_data for k, v in d.items()}
    print(document_data)
    return render_template('all-store-wise-data.html', document=document_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
