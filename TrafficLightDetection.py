
# TRAFFIC LIGHT DETECTION CODE:

import cv2
import numpy as np
from picamera2 import Picamera2

# Load the pre-trained CNN model for traffic light detection
model_path = "path_to_pretrained_model"  # Change this to the path of your pre-trained model
net = cv2.dnn.readNet(model_path)

# Function to check for traffic light in the image using the CNN model
def check_traffic(frame):
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])
            if class_id == 1:  # Traffic light class ID
                print("Traffic light detected")
                # Extract bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Processed Image", frame)
    cv2.waitKey(1)

# Main function
if __name__ == '__main__':
    camera = Picamera2()
    camera.start()
    try:
        while True:
            frame = camera.capture_array()  # Directly capture the frame
            check_traffic(frame)
    finally:
        camera.stop()
        cv2.destroyAllWindows()

