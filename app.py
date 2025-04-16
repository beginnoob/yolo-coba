import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from utils import process_image, process_video, process_camera
import cv2
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = 'static/models/best.pt'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        # Check which button was clicked
        if 'upload_photo' in request.form:
            return redirect(url_for('upload_photo'))
        elif 'upload_video' in request.form:
            return redirect(url_for('upload_video'))
        elif 'open_camera' in request.form:
            return redirect(url_for('camera'))
    
    return render_template('scan.html')

@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            result_image, prediction = process_image(filepath, app.config['MODEL_PATH'], hands)
            
            # Save the processed image
            processed_filename = 'processed_' + filename
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, result_image)
            
            session['result_image'] = processed_filename
            session['prediction'] = prediction
            
            return redirect(url_for('result'))
    
    return render_template('upload_photo.html')

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the video
            output_filename = process_video(filepath, app.config['MODEL_PATH'], hands)
            
            session['result_video'] = output_filename
            return redirect(url_for('result'))
    
    return render_template('upload_video.html')

@app.route('/camera')
def camera():
    # Process camera feed
    output_filename = process_camera(app.config['MODEL_PATH'], hands)
    
    if output_filename:
        session['result_image'] = output_filename
        return redirect(url_for('result'))
    
    return render_template('camera.html')

@app.route('/result')
def result():
    result_image = session.get('result_image', None)
    result_video = session.get('result_video', None)
    prediction = session.get('prediction', None)
    
    return render_template('result.html', 
                         result_image=result_image,
                         result_video=result_video,
                         prediction=prediction)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)