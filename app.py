from flask import Flask, request, send_file
import os
import threading
import main  # Importing your main.py script

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return send_file('./templates/index.html')

@app.route('/upload')
def upload():
    return send_file('./templates/photo.html')

@app.route('/suggestions')
def suggestions():
    return send_file('./uploads/camera_suggestions.txt')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'panoramaFile' not in request.files:
        return '<p class="text-red-600">No file part</p>'
    file = request.files['panoramaFile']
    if file.filename == '':
        return '<p class="text-red-600">No selected file</p>'
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'panoPhoto.png')
        file.save(filename)

        # Run main.py in a separate thread
        thread = threading.Thread(target=main.process_img)  # Assuming process_img is the function in main.py
        thread.start()

        return '<p class="text-green-600">File uploaded successfully! Image processing started.</p>'

@app.route('/toronto.jpg')
def serve_background():
    return send_file('toronto.jpg')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)