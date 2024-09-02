from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2 as cv
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load YOLO model and class names globally
if not os.path.exists('yolov3.cfg') or not os.path.exists('yolov3.weights'):
    print("Configuration or weights file not found!")
else:
    yolo_net = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')

class_names = []
with open('coco.names', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def process_image(image_path):
    image = cv.imread(image_path)
    img_height, img_width = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    output_layer_names = yolo_net.getUnconnectedOutLayersNames()
    network_output = yolo_net.forward(output_layer_names)
    
    class_ids, confidences, bounding_boxes = [], [], []
    for output in network_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                w = int(detection[2] * img_width)
                h = int(detection[3] * img_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bounding_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        indices = indices.flatten()

    for i in indices:
        if isinstance(i, list) or isinstance(i, tuple):
            i = i[0]
        box = bounding_boxes[i]
        x, y, w, h = box
        label = str(class_names[class_ids[i]])
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    cv.imwrite(output_path, image)
    print(f"Image saved at: {output_path}")
    return output_path

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output_image_path = process_image(filepath)
            return redirect(url_for('uploaded_file', filename='output.jpg'))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
