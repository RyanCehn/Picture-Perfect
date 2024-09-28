from flask import Flask, request, send_file
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/upload')
def upload():
    return send_file('photo.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'panoramaFile' not in request.files:
        return '<p class="text-red-600">No file part</p>'
    file = request.files['panoramaFile']
    if file.filename == '':
        return '<p class="text-red-600">No selected file</p>'
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return f'<p class="text-green-600">File {file.filename} uploaded successfully!</p>'

@app.route('/toronto.jpg')
def serve_background():
    return send_file('toronto.jpg')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)