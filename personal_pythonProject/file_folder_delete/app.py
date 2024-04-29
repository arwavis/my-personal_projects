from flask import Flask, render_template, request, redirect, url_for
import os

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
        remove_contents(folder_path)
        return redirect(url_for('index'))
    except Exception as e:
        return str(e)


def remove_contents(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            if os.path.isdir(dir_path) and not os.path.islink(dir_path):
                os.rmdir(dir_path)

    return 'Contents of the directory deleted.'


if __name__ == '__main__':
    app.run(debug=True)
