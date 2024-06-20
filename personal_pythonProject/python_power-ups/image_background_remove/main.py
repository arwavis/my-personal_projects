from rembg import remove
from PIL import Image
import sys
import os


def removebg(inputPath):
    # Get the file extension
    file_extension = os.path.splitext(inputPath)[1][1:].lower()
    # Validate file type
    if file_extension not in ['jpg', 'jpeg', 'png']:
        print(f"Warning: Unsupported file type '{file_extension}'. Supported file types are: jpg, jpeg, png.")
        sys.exit(1)
    try:
        # Open the image and remove its background
        originalImage = Image.open(inputPath)
        imageWithoutBg = remove(originalImage)
        # Save the new image with a .png extension
        outputPath = os.path.splitext(inputPath)[0] + '.png'
        imageWithoutBg.save(outputPath)
        print("Background removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Validate the number of arguments
    if len(sys.argv) < 2:
        print("Usage: python removebg.py <input_path>")
        sys.exit(1)
    # Run the removebg function
    removebg(sys.argv[1])

"""run this code from the command line using python main.py input.jpg, replacing input.jpg with the path to the image 
you want to remove the background from."""
