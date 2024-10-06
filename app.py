from flask import Flask, request, render_template, Response, url_for
from PIL import Image
import torch
import io
import shutil
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='assets')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'assets/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    img = Image.open(file_path)

    # Perform object detection
    results = model(img)
    
    results_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    results.save(save_dir=results_path)  # Save new results

    return render_template('result.html', results=results_path)

def gen_frames():
    cap = cv2.VideoCapture(0)  # Capture video from the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Perform object detection on the frame
            results = model(frame)
            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', results.render()[0])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
