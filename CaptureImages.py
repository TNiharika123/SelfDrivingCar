# CAPTURE THROUGH CAMERA:


if __name__ == '__main__':
    camera = Picamera2()
    camera.start()
    try:
        while True:
            image = camera.capture_array()  # Directly capture the image
            process_image(image)
    finally:
        camera.stop()
        cv2.destroyAllWindows()



