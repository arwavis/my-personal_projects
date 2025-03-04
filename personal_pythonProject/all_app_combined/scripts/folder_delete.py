import os
import shutil

folder_path = '/Users/aravindv/Downloads/to_be_deleted'


def delete_all_files():
    """Deletes all files in the specified folder."""
    try:
        shutil.rmtree(folder_path)  # Remove the folder
        os.makedirs(folder_path)  # Recreate an empty folder
        return "All files deleted successfully!"
    except Exception as e:
        return f"Error: {e}"
