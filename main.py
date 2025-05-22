import cv2
import numpy as np
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import os
import glob

def enhance_image_for_barcode(image):
    """Apply image preprocessing to enhance barcode detection"""
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Try different preprocessing methods if needed
    preprocessing_methods = [
        ("original", gray),
        ("adaptive_threshold", thresh),
        ("gaussian_blur", cv2.GaussianBlur(gray, (5, 5), 0)),
        ("sharpen", cv2.filter2D(gray, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])))
    ]
    
    return preprocessing_methods

def detect_and_decode_barcode(image_path, save_result=True, show_result=True):
    """Detect and decode barcodes from an image file"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Make a copy for drawing results
    result_image = image.copy()
    
    # Get filename without extension for saving results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Apply different preprocessing methods for better detection
    preprocessing_methods = enhance_image_for_barcode(image)
    
    # Initialize variables to track if we found any barcodes
    found_barcode = False
    all_barcodes = []
    
    # Try each preprocessing method until we find barcodes
    for method_name, processed_img in preprocessing_methods:
        # Detect barcodes in the processed image
        barcodes = decode(processed_img)
        
        if barcodes:
            found_barcode = True
            print(f"Found {len(barcodes)} barcode(s) using {method_name} method")
            
            # Loop over detected barcodes
            for barcode in barcodes:
                # Extract barcode data and type
                barcode_data = barcode.data.decode("utf-8")
                barcode_type = barcode.type
                
                # Avoid duplicates
                if (barcode_data, barcode_type) not in all_barcodes:
                    all_barcodes.append((barcode_data, barcode_type))
                
                # Print barcode data and type
                print(f"Barcode Data: {barcode_data}")
                print(f"Barcode Type: {barcode_type}")
                
                # Draw a rectangle around the barcode
                (x, y, w, h) = barcode.rect
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Put barcode data and type on the image
                text = f"{barcode_data} ({barcode_type})"
                y_offset = max(y - 10, 10)  # Ensure text is within image bounds
                cv2.putText(result_image, text, (x, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if not found_barcode:
        print(f"No barcodes found in {image_path}")
        # Try more aggressive processing as a last resort
        # Increase contrast
        contrast_img = cv2.convertScaleAbs(image, alpha=2.0, beta=0)
        gray_contrast = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray_contrast)
        
        if barcodes:
            print(f"Found {len(barcodes)} barcode(s) using high contrast method")
            for barcode in barcodes:
                # Extract barcode data and type
                barcode_data = barcode.data.decode("utf-8")
                barcode_type = barcode.type
                all_barcodes.append((barcode_data, barcode_type))
                print(f"Barcode Data: {barcode_data}")
                print(f"Barcode Type: {barcode_type}")
    
    # Save or show results
    if save_result and found_barcode:
        # Convert image from BGR to RGB (Matplotlib uses RGB)
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(result_image_rgb)
        plt.axis('off')
        plt.title(f"Detected Barcodes: {len(all_barcodes)}")
        
        # Save the result
        output_filename = f"{base_name}_detected.png"
        plt.savefig(output_filename)
        print(f"Result saved as {output_filename}")
        
        if show_result:
            plt.show()
    
    return all_barcodes

def process_multiple_images(image_paths):
    """Process multiple images for barcode detection"""
    results = {}
    
    for image_path in image_paths:
        print(f"\nProcessing {image_path}...")
        barcodes = detect_and_decode_barcode(image_path)
        results[image_path] = barcodes
    
    return results

# Example usage
if __name__ == "__main__":
    # Process a single image
    # detect_and_decode_barcode("IMG_6759.jpeg")
    
    # Process all images in a directory
    image_paths = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
    if not image_paths:
        print("No image files found in the current directory")
        # If no images found, use the hardcoded filename
        image_paths = ["IMG_6759.jpeg"]
        
    process_multiple_images(image_paths)