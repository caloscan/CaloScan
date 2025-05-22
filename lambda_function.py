import cv2
import numpy as np
from pyzbar.pyzbar import decode
import json
import base64
import boto3
import io

def detect_barcode(image_bytes):
    """Process image bytes to detect barcodes"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocessing methods - ordered by reliability
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    methods = [
        ("original", gray),
        ("adaptive_threshold", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)),
        ("gaussian_blur", cv2.GaussianBlur(gray, (5, 5), 0))
    ]
    
    # Track detected barcodes and their frequency
    barcode_counts = {}
    barcode_objects = {}
    
    # Process methods in order of reliability
    for method_name, processed_img in methods:
        barcodes = decode(processed_img)
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            key = (data, barcode_type)
            
            # Count frequency across methods
            barcode_counts[key] = barcode_counts.get(key, 0) + 1
            
            # Store the barcode object 
            if key not in barcode_objects:
                barcode_objects[key] = barcode
    
    # No barcodes found
    if not barcode_counts:
        return None
    
    # Get the most frequently detected barcode
    best_barcode = max(barcode_counts.items(), key=lambda x: x[1])
    data, barcode_type = best_barcode[0]
    confidence = best_barcode[1] / len(methods)
    
    return {
        'data': data,
        'type': barcode_type,
        'confidence': confidence
    }

def get_image_from_s3(bucket, key):
    """Get image bytes from S3"""
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        # Check if the event is from API Gateway (base64 encoded image)
        if 'body' in event and event.get('isBase64Encoded', False):
            # Decode base64 image
            image_bytes = base64.b64decode(event['body'])
        
        # Check if the event is from S3
        elif 'Records' in event and event['Records'][0].get('eventSource') == 'aws:s3':
            record = event['Records'][0]['s3']
            bucket = record['bucket']['name']
            key = record['object']['key']
            image_bytes = get_image_from_s3(bucket, key)
        
        # Direct binary content (for testing)
        elif 'image' in event and isinstance(event['image'], str):
            image_bytes = base64.b64decode(event['image'])
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid input format'})
            }
        
        # Detect barcode
        result = detect_barcode(image_bytes)
        
        if result:
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "success": true,
                    "data": {
                        "barcodeValue": result['data'],
                        "barcodeType": result['type'],
                        "confidence": result['confidence']
                    },
                    "requestId": context.awsRequestId
                })
            }
        
        # No barcode found
        else:
            return {
                "statusCode": 404,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "success": false,
                    "error": {
                        "code": "BARCODE_NOT_FOUND",
                        "message": "No barcode could be detected in the provided image"
                    },
                    "requestId": context.awsRequestId
                })
            }
            
    except Exception as e:
        # Handle real errors (not just missing barcodes)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "success": false,
                "error": {
                    "code": "PROCESSING_ERROR",
                    "message": "An error occurred while processing the image"
                },
                "requestId": context.awsRequestId
            })
        }