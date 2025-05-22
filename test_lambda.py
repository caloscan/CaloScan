# Create a test_locally.py file
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from lambda_function import detect_barcode  # Import your Lambda function

# Test with a local image
def test_with_file(img):
    with open(img, 'rb') as f:
        image_bytes = f.read()
    
    # Call your barcode detection function
    result = detect_barcode(image_bytes)
    print("Result:", result)

if __name__ == "__main__":
    test_with_file('IMG_6760.jpeg')
    test_with_file('IMG_6761.jpeg')
    test_with_file('IMG_6764.jpeg')
    test_with_file('IMG_6759.jpeg')