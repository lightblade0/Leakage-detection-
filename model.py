from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)
model = YOLO("best.pt")  # Load YOLO model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])  # Decode base64 image
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model.predict(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    output_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image": f"data:image/jpeg;base64,{output_image}"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
