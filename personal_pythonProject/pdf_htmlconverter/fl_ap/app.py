from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pdfplumber

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pdf_to_html(pdf_file_path, html_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    with open(html_file_path, 'w', encoding='utf-8') as html_file:
        html_file.write("<html><body>")
        html_file.write("<pre>{}</pre>".format(text))
        html_file.write("</body></html>")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            html_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.html')
            pdf_to_html(pdf_path, html_path)
            return redirect(url_for('uploaded_file', filename='output.html'))
    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
