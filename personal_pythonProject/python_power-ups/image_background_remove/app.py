import os
from flask import Flask, render_template, request, send_file
from rembg import remove
from PIL import Image
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def remove_background():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = str(uuid.uuid4()) + '.' + file.filename.split('.')[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            originalImage = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imageWithoutBg = remove(originalImage)
            outputFilename = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)
            imageWithoutBg.save(outputFilename, format='PNG')
            return render_template('index.html', filename=os.path.basename(outputFilename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
