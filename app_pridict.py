import time
import os
import cv2
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pynput import keyboard
import csv 
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import threading
import pyodbc  # Add pyodbc import for database connection
import datetime  # Add datetime import for date handling
import pandas as pd

scanned_data = []  # To store the scanned data
statuses = []  # Initialize statuses list
partno = ""  # Initialize partno
second_partno = ""  # Initialize second_partno
image_path1 = ""  # Initialize image_path1
image_path2 = ""  # Initialize image_path2
app = Flask(__name__)  # Initialize Flask app
model_type = "WITH_NBRR"  # Default model type
base_path = r"C:\Users\tv239\Downloads\EOL\\"

def eclip():
    model4 = YOLO(base_path + r"TRAINED_MODEL\eclips\eclips 8s 100ep  best (24)04_12_24.pt") #E CLIP WEIGHTS 
    results3 = model4.predict(source=image_path1, save=True, conf=0.2)
    
    #for re in results3:
    #re.show()
    
    im1 = base_path + r"runs\detect\predict"  # predn folder
    filesim1 = os.listdir(im1)
    image_filename = filesim1[0]  # img name inside predn folder
   
    im1_copy_path = os.path.join(im1, image_filename)    #  predn img path

    #print(im1_copy_path)

    save_folder = base_path + r"prediction images folder"  # folder in which img to be saved
    
    new_image_name = "eclip.jpg"
    renamed_img_path = os.path.join(save_folder, new_image_name) 

    shutil.move(im1_copy_path, renamed_img_path)  #  move predn img in this new folder

    # Define the folder path you want to delete (e.g., 'runs/predict/exp')
    folder_to_delete = im1  # deleting predn folder

    # Check if the folder exists, then delete it
    if os.path.exists(folder_to_delete):
        shutil.rmtree(folder_to_delete)
        print(f"Folder '{folder_to_delete}' has been deleted.")
    else:
        print(f"Folder '{folder_to_delete}' does not exist.")
    
    class_names = model4.names  # This should give you a list or dictionary of class names (index -> class)
    print("class names: ", class_names)
    eclip_class_name = "eclip-yes"  # Replace with your actual class name for dataplate

    eclip_found = False  # Flag to check if a dataplate is detected
    eclipcount = 0
    for result in results3:
        for box in result.boxes:
            print("BOX: ", box)
            class_id = int(box.cls[0])  # Get the class ID (index)
            print("CLASS ID: ", class_id)
            class_name = class_names[class_id]  # Convert the ID to class name
            
            if class_name == eclip_class_name:
                eclip_found = True
                print(f"eclip detected with confidence: {box.conf[0]}")  # Print confidence score too if needed
                eclipcount += 1
    if eclipcount >= 1:
        print("eclip present.")
        statuses.append("OK")
    else:
        print("eclip not found.")
        statuses.append("NOT OK")
    print("status update for eclip: ", statuses)

def drivescrew(image_path1):
    model6 = YOLO(base_path + r"TRAINED_MODEL\drive screw+daatplate\best (27).pt")  # Replace with the path to your trained model weights
    model7 = YOLO(base_path + r"TRAINED_MODEL\drive screw+daatplate\drivescrew yolov11 best (32).pt")
    image3 = cv2.imread(image_path1)
    results6 = model6.predict(source=image_path1, save=False, conf=0.3)
    for idx, result in enumerate(results6):
        for box_id, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get the coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer

            cropped_image = image3[y1:y2, x1:x2]
            output_path = f'cropped_image for drive screw_{idx}_{box_id}.jpg'
            cv2.imwrite(output_path, cropped_image)
            print(f"Cropped image saved as {output_path}")
            #result.show()

    results7 = model7.predict(source=cropped_image, save=True, conf=0.4)
    #for res in results7:
        #res.show()

    class_names = model7.names  # This should give you a list or dictionary of class names (index -> class)
    print("class names: ", class_names)
    drivescrew_class_name = "drive-screw-yes"  # Replace with your actual class name for dataplate

    drivescrew_found = False  # Flag to check if a dataplate is detected
    drivecount = 0
    for result in results7:
        for box in result.boxes:
            print("BOX: ", box)
            class_id = int(box.cls[0])  # Get the class ID (index)
            print("CLASS ID: ", class_id)
            class_name = class_names[class_id]  # Convert the ID to class name
            
            if class_name == drivescrew_class_name:
                drivescrew_found = True
                print(f"Drivescrew detected with confidence: {box.conf[0]}")  # Print confidence score too if needed
                drivecount += 1
    if drivecount >= 2:
        print("Drivescrew present.")
        statuses.append("OK")
    else:
        print("Drivescrew not found.")
        statuses.append("NOT OK")
    print("status update for drive: ", statuses)
    
    im4 = base_path + r"runs\detect\predict"  # predn folder
    filesim1 = os.listdir(im4)
    image_filename = filesim1[0]  # img name inside predn folder
    
    im1_copy_path = os.path.join(im4, image_filename)    #  predn img path

    #print(im1_copy_path)

    save_folder = base_path + r"prediction images folder"  # folder in which img to be saved
    
    new_image_name = "drivescrew.jpg"
    renamed_img_path = os.path.join(save_folder, new_image_name) 

    shutil.move(im1_copy_path, renamed_img_path)  #  move predn img in this new folder

    folder_to_delete = im4  # deleting predn folder

    if os.path.exists(folder_to_delete):
        shutil.rmtree(folder_to_delete)
        print(f"Folder '{folder_to_delete}' has been deleted.")
    else:
        print(f"Folder '{folder_to_delete}' does not exist.")

def nbbr(image_path1):
    model8 = YOLO(base_path + r"TRAINED_MODEL\nbbr\best.pt nbbr yolov8s 100ep.pt")
    result9 = model8.predict(source=image_path1, save=True, conf=0.7)
    
    im1 = base_path + r"runs\detect\predict"  # predn folder
    filesim1 = os.listdir(im1)
    image_filename = filesim1[0]  # img name inside predn folder

    im1_copy_path = os.path.join(im1, image_filename)    #  predn img path

    #print(im1_copy_path)

    save_folder = base_path + r"prediction images folder"  # folder in which img to be saved
    
    new_image_name = "nbbr.jpg"
    renamed_img_path = os.path.join(save_folder, new_image_name) 

    shutil.move(im1_copy_path, renamed_img_path)  #  move predn img in this new path

    folder_to_delete = im1  # deleting predn folder

    if os.path.exists(folder_to_delete):
        shutil.rmtree(folder_to_delete)
        print(f"Folder '{folder_to_delete}' has been deleted.")
    else:
        print(f"Folder '{folder_to_delete}' does not exist.")

    class_names = model8.names  # This should give you a list or dictionary of class names (index -> class)
    print("class names: ", class_names)
    nbbr_class_name = "nbrr-present"  # Replace with your actual class name for dataplate

    nbbr_found = False  # Flag to check if a dataplate is detected
    nbbrcount = 0
    for result in result9:
        for box in result.boxes:
            print("BOX: ", box)
            class_id = int(box.cls[0])  # Get the class ID (index)
            print("CLASS ID: ", class_id)
            class_name = class_names[class_id]  # Convert the ID to class name
            
            if class_name == nbbr_class_name:
                nbbr_found = True
               #print(f"nbbr detected with confidence: {box.conf[0]}")  # Print confidence score too if needed
                nbbrcount += 1
    if nbbrcount >= 1:
        print("nbbr present.")
        statuses.append("OK")
        statuses.append("OK")
    else:
        print("nbbr not found.")
        statuses.append("NOT OK")
        statuses.append("NOT OK")
    print("status update for nbbr: ", statuses)

def cameracapture(save_path):
    global image_path1, image_path2, image_path
    import cv2
    import time
    import os

    image_path2 = os.path.join(save_path, "Camera_top_image.jpg")
    image_path1 = os.path.join(save_path, "camera_side_image.jpg")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap1 = cv2.VideoCapture(0)   #camera 3 is up  #1 is side up
    cap2 = cv2.VideoCapture(1) #razor

    if not cap1.isOpened():
        print("Error: Could not open the camera.")
        exit()

    start_time = time.time()

    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 means manual focus   #35 is 115 and 34 is 120. for 1 its 120. other is 110
    cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    cap1.set(cv2.CAP_PROP_FOCUS, 100)  # Adjust the focus value   # value 78 works perfectly. 85 also good
    cap2.set(cv2.CAP_PROP_FOCUS, 85)
    cap1.set(3, 1280)  # Set width
    cap2.set(3, 1280)

    capture_after_seconds = 10  # Time delay for capturing image

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Error: Could not read frame from one or both cameras.")
            break
        elapsed_time = time.time() - start_time
        if elapsed_time >= capture_after_seconds:
            cv2.imwrite(image_path1, frame1)  # Save frame from camera 1
            cv2.imwrite(image_path2, frame2)  # Save frame from camera 2
            print(f"Image from camera 1 saved at {image_path1}")
            print(f"Image from camera 2 saved at {image_path2}")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cv2.destroyAllWindows()

scanned_data = []  # List to hold scanned data
file_path = base_path + r"scanned_text_1.txt"

def get_db_connection():
    """Establish and return a database connection."""
    server = '(localdb)\\MSSQLLocalDB'   
    #server = ''
    #user= ''
    #password = ''
    database = 'EOL_CHECKING'
    #database = 'MitsProjectDbNew'
   
    #database = ''
    #server = ''
    driver = '{ODBC Driver 17 for SQL Server}'  # Adjust the driver according to your SQL Server version

    try:
        #conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={user};PWD={password}')
        conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;')
        return conn
    except pyodbc.Error as e:
        print("Database connection error:", e)
        return None

def insert_data_to_db(data):
    """Insert data into the database."""
    conn = get_db_connection()
    if conn is None:
        print('Database connection error')
        return 'Database connection error'

    table_name = 'eol_table'

    cursor = conn.cursor()
    try:
        # Ensure data is in the correct format
        formatted_data = [(row[0], row[1], row[2], row[3], row[4], row[5], row[6], 
                           row[7], row[8], row[9]) for row in data]
        print("Formatted data to insert:", formatted_data)  # Debugging statement
        cursor.executemany(f"INSERT INTO {table_name} (part_no, serial_number, eclip, drive_screw, nb, rr, time, eclip_path, dataplate_screw_path, nbbr_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", formatted_data)
        conn.commit()
        print("Data inserted successfully")  # Debugging statement
    except pyodbc.Error as e:
        print("Error inserting data:", e)
    finally:
        cursor.close()
        conn.close()

@app.route('/set_model_type')
def set_model_type():
    global model_type
    model_type = request.args.get('model_type', 'WITH_NBRR')
    return jsonify(success=True)

def main_function():
    global image_path, image_path1, image_path2, c1, statuses, new_data
    st2 = time.time()
    new_data = [] #excel list containing all values of each defect
   
    #save_folder = base_path + r"cameracapture_folder"
    #cameracapture(save_folder)
   
    #image_path1 = base_path + r"eclip1.png"
    image_path1 = base_path + r"eclip_miss.jpg"
    #image_path2 = base_path + r"top1.jpg"
    image_path2 = base_path + r"without_nbbr.jpg"
  
    eclip()  # Remove the argument
    drivescrew(image_path2)
    
    if model_type == "WITH_NBRR":
        nbbr(image_path2)
        nb_status = statuses[2]  # Assuming nb status is at index 2
        rr_status = statuses[3]  # Assuming rr status is at index 3
    else:
        nb_status = None
        rr_status = None
     
    print("STATUSES: ", statuses)
    from datetime import datetime
 
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
    print("Formatted datetime:", formatted_datetime)  # Debugging statement
   
    timestamp = current_datetime.strftime("%Y%m%d_%H%M")
    eclip_filename = f"eclip_{timestamp}.jpg"
    drivescrew_filename = f"drivescrew_{timestamp}.jpg"
    nbbr_filename = f"nbbr_{timestamp}.jpg"
   
    new_data.append([partno, second_partno, statuses[0], statuses[1], nb_status, rr_status, formatted_datetime, eclip_filename, drivescrew_filename, nbbr_filename])
    print("New data to insert:", new_data)  # Debugging statement
    insert_data_to_db(new_data)
    print("NEW DATA: ", new_data)
   
    if "NOT OK" in statuses:
        print("DEFECT")
       
        bin_output = base_path + r"status_output_file.txt"
        with open(bin_output, "w") as bo:
            bo.write('1')
            
        import winsound
        alarm_sound = base_path + r"slot-machine-payout-alarm-1996.wav"
        winsound.PlaySound(alarm_sound, winsound.SND_FILENAME | winsound.SND_LOOP)
        time.sleep(0.5)  # Let the sound play for 5 seconds
        winsound.PlaySound(None, winsound.SND_PURGE)  # Stop the sound
        
    else:
        print("NO DEFECT")
        bin_output = base_path + r"status_output_file.txt"
        with open(bin_output, "w") as bo:
            bo.write('0')
    time.sleep(10)
    statuses = []  # statuses list cleared
    new_data = []
    
    source_folder = base_path + r"prediction images folder"
    destination_folder = base_path + r"static\images"
    destination_folder_2_quality = base_path + r"static\quality_check_prediction_folder"

    for filename in os.listdir(source_folder):
        source_image_path = os.path.join(source_folder, filename)
        if os.path.isfile(source_image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_extension = os.path.splitext(filename)[1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            new_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{file_extension}"
            destination_image_path = os.path.join(destination_folder, new_filename)
            destination_quality_path = os.path.join(destination_folder_2_quality, new_filename)
            shutil.copy(source_image_path, destination_image_path)
            shutil.copy(source_image_path, destination_quality_path)
            print(f"Copied {filename} to {destination_folder} and {destination_folder_2_quality}")

    print("All images copied successfully.")
    
    et2 = time.time()
    print("ESTIMATED OVERALL TIME: ", et2 - st2)

    c1 = 0
    print("Resetting c1 to 0")
    
    folder_path = base_path + r"prediction images folder"
    for file_name in os.listdir(folder_path):
        del_filepath = os.path.join(folder_path, file_name)
        if os.path.isfile(del_filepath):
            os.remove(del_filepath)
            print("PREDN IMAGES FOLDER DELETED")

def is_file_empty(file_path):
    """Check if the file is empty."""
    return os.stat(file_path).st_size == 0

def on_press(key):
    """Handle keyboard events."""
    global scanned_data
    print("Inside on_press")

    if hasattr(key, 'char') and key.char is not None:
        scanned_data.append(key.char)
        print("Character added to list:", scanned_data)
    
    elif key == keyboard.Key.enter:
        data = ''.join(scanned_data)
        with open(file_path, 'a') as f:
            print("Writing data to file:", scanned_data)
            f.write(data + '\n')

        print("Data written to file:", scanned_data)
        scanned_data.clear()
        print("Data cleared after saving")

def keyboard_listener():
    """Run the keyboard listener in a separate thread."""
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def monitor_file_and_run_model():
    global partno, second_partno
    """Continuously check if file has contents and run model if it does."""
    while True:
        if not is_file_empty(file_path):
            print("File has contents, running the main function...")
            with open(file_path, 'r') as file:
                line = file.readline().strip()
                print("Read line from file:", line)  # Debugging statement
                values = line.split(',')
                print("Split values:", values)  # Debugging statement
                if len(values) >= 3:
                    partno = values[2]
                    second_partno = values[1]
                    print("Part number:", partno)  # Debugging statement
                    print("Second part number:", second_partno)  # Debugging statement
                else:
                    print("Error: Not enough values in the line.")
                    continue
            with open(file_path, "w") as a:
                a.write('')

            bin_output = base_path + r"status_output_file.txt"
            with open(bin_output, "w") as bo:
                bo.write(' ')
            main_thread = threading.Thread(target=main_function)
            main_thread.start()

def run_flask():
    app.run(debug=True, use_reloader=False)

@app.route('/')
def home():
    global statuses, model_type
    data = fetch_data_from_db()
    return render_template('index.html', data=data, enumerate=enumerate, model_type=model_type)

@app.route('/download_excel')
def download_excel():
    data = fetch_data_from_db()
    if not data:
        return "No data available to download", 404

    df = pd.DataFrame(data)
    excel_path = base_path + r"eol_data.xlsx"
    df.to_excel(excel_path, index=False)

    directory = os.path.dirname(excel_path)
    filename = os.path.basename(excel_path)
    return send_from_directory(directory, filename, as_attachment=True)

def fetch_data_from_db():
    conn = get_db_connection()
    if conn is None:
        return []

    table_name = 'eol_table'
    cursor = conn.cursor()
    try:
        query = f"""
        SELECT [part_no], [serial_number], [eclip], [drive_screw], [nb], [rr], 
        FORMAT([time], 'dd/MM/yyyy HH:mm:ss') as [time], [eclip_path], [dataplate_screw_path], [nbbr_path]
        FROM {table_name}
        WHERE [time] >= DATEADD(day, -5, GETDATE())
        ORDER BY [time] DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]
        return data
    except pyodbc.Error as e:
        print("Error fetching data:", e)
        return []
    finally:
        cursor.close()
        conn.close()

@app.route('/view_images/<serial_number>')
def view_images(serial_number):
    data = fetch_data_from_db()
    for row in data:
        if row['serial_number'] == serial_number:
            return render_template('view_images.html', row=row)
    return "Serial number not found", 404

import webbrowser

def main():
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    # Open the browser to the home page
    webbrowser.open('http://127.0.0.1:5000/')
    monitor_file_and_run_model()

def MAIN_FUNCTION():
    global scanned_data, l, output_csv, statuses
    scanned_data = []
    statuses = []
    l = []
    output_csv = r'C:\Users\tv239\Downloads\intensity.csv'
    main()

MAIN_FUNCTION()
