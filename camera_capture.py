import cv2
import time
import os
from datetime import datetime
 
# Global variables
save_path = r"C:\Users\tv239\Downloads\FINAL_CODE\captured images folder"
 
# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)
 
# Open video capture for the webcam (0 for default camera) 110
cap1 = cv2.VideoCapture(0)   # camera 3 is up  # 1 is side up
 
# Check if the camera opened successfully
if not cap1.isOpened():
    print("Error: Could not open the camera.")
    exit()
 
start_time = time.time()
 
# Disable auto-focus and manually set focus for the webcam
cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 means manual focus
cap1.set(cv2.CAP_PROP_FOCUS, 78)     # Adjust the focus value
cap1.set(3, 1280)                    # Set width
 
capture_after_seconds = 5  # Time delay for capturing image
 
while True:
    # Read frame from the webcam
    ret1, frame1 = cap1.read()
 
    if not ret1:
        print("Error: Could not read frame from the camera.")
        break
    cv2.imshow('Live Camera Feed', frame1)
 
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
 
    # Capture image after the specified time
    if elapsed_time >= capture_after_seconds:
        # Generate image path with current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
 
        #image_path1 = os.path.join(save_path, f"vecv_nbbr_OK_{current_time}.jpg")
        image_path1 = os.path.join(save_path, f"crimping_miss{current_time}.jpg")
       #image_path1 = os.path.join(save_path, f"locknut_miss{current_time}.jpg")
       # image_path1 = os.path.join(save_path, f"one_drive_screw_miss{current_time}.jpg")
        cv2.imwrite(image_path1, frame1)  # Save frame from the camera
 
        print(f"Image from the camera saved at {image_path1}")
        break  # Exit the loop after capturing the image
 
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release the video capture object and close windows
cap1.release()
cv2.destroyAllWindows()
 