from flask import Flask, render_template, request, redirect, url_for
import os
import shutil

app = Flask(__name__)
folder_path = '/Users/aravindv/Downloads/to_be_deleted'  # Path to the directory


@app.route('/')
def index():
    file_infos = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_infos.append({'name': file, 'path': file_path})
    return render_template('index.html', files=file_infos)


@app.route('/delete', methods=['POST'])
def delete_files():
    try:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        return render_template('index.html', files=[], message='All files in the folder have been deleted successfully!')
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
