# OBJECT DETECTION
from picamera2 import Picamera2
import numpy as np
import cv2 as cv
import RPi.GPIO as GPIO          
from time import sleep

in1 = 16
in2 = 18
in3 = 11
in4 = 13
ena = 22
enb = 15

temp1 = 1

GPIO.setmode(GPIO.BOARD) 
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(ena, GPIO.OUT)
GPIO.setup(enb, GPIO.OUT)

p1 = GPIO.PWM(ena, 5000)
p2 = GPIO.PWM(enb, 5000)

GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)

p1.start(25)
p2.start(25)
p1.ChangeDutyCycle(75)
p2.ChangeDutyCycle(75)

whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

def measure_distance(image, object_size_pixel):
    # Assuming you have a known focal length of the camera in pixels
    focal_length_pixels = 500  # Example: Focal length is 500 pixels
    
    # Assuming you have a known actual size of the object in meters
    object_size_meters = 0.1  # Example: Object size is 0.1 meters (10 cm)
    
    # Calculate distance using the basic formula: distance = (focal_length * object_size) / actual_size
    distance = (focal_length_pixels * object_size_meters) / object_size_pixel
    distance=distance*100
    
    return distance



#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
## Model Files
modelConfiguration = "/home/pi/YoloV3/yolov3.cfg"
modelWeights = "/home/pi/YoloV3/yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def stop(image, object_size_str):
    # Parse the object size string to extract width and height
    object_size = object_size_str.split("x")
    object_width_pixels = int(object_size[0])
    object_height_pixels = int(object_size[1])

    # Calculate the average size of the object (assuming it's the average of width and height)
    object_size_pixel = (object_width_pixels + object_height_pixels) / 2

    distance = measure_distance(image, object_size_pixel)
    if distance <= 25:  # Check if the car is within 30 cm of the object
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        print("Object detected at distance:", distance, "cm. Stopping the car.")
    else:
        print("Object detected at distance:", distance, "cm. Car is not close enough to stop.")       
        
def forward():
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    print("No object detected: Moving forward")

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    object_names = []
    object_sizes = []  # Store sizes of detected objects
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                object_names.append(classNames[classId].upper())
                object_sizes.append(f"{w}x{h}")

    print(len(bbox))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Convert img to cv::UMat
    img_umat = cv.UMat(img)
    print("Detected objects:", object_names)

    if len(object_names) == 0:
        forward()
        print("No object detected: Moving forward")
    else:
        for i in indices:

            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            print("x,y,w,h", x, y, w, h)
            object_size_str = object_sizes[i]

            cv.rectangle(img_umat, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print("size of object")
            # Put text with class name and confidence level
            cv.putText(img_umat, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            object_size_str = object_sizes[i]
            stop(img, object_size_str)
            print("Object detected:", object_names[i].upper())

    # Convert img_umat back to numpy.ndarray
    img = img_umat.get()
    

def process_image(image):
    image = image[:, :, :3]
    
    blob = cv.dnn.blobFromImage(image, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    
    outputNames = [layersNames[i- 1] for i in net.getUnconnectedOutLayers()]
    
    outputs = net.forward(outputNames)
    
    
    findObjects(outputs, image)
    cv.imshow('Image', image)
    cv.waitKey(1)

camera = Picamera2()  # Initialize PiCamera
camera.start()
try:
    while True:
        image = np.empty((480, 640, 3), dtype=np.uint8)  # Create an empty numpy array to hold the image
        image = camera.capture_array()   # Capture image in BGR format
        
        process_image(image)
finally:
    camera.stop()  # Close the PiCamera instance
    cv.destroyAllWindows()
