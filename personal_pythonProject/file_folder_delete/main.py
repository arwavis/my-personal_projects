import os

folder_path = '/Users/aravindv/Downloads/to_be_deleted'


def remove_contents(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


try:
    remove_contents(folder_path)
    print('Contents of the directory deleted.')
except Exception as e:
    print('An error occurred while deleting the contents of the directory:', str(e))
