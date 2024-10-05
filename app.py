from flask import Flask, request, render_template, Response, url_for
from PIL import Image
import torch
import io
import shutil
import cv2
import os

# Configure Flask to use 'assets' as the static folder
app = Flask(__name__, static_folder='assets')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Perform object detection
    results = model(img)
    
    result_dir = 'assets/results'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)  # Clear old results
    results.save(save_dir=result_dir)  # Save new results

    return render_template('result.html')

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
    app.run(host='0.0.0.0', port=5000, debug=True)
