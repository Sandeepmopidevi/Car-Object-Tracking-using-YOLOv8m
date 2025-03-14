import cv2
import torch
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize Tkinter window
root = tk.Tk()
root.title("Car Detection & Tracking")
root.geometry("900x700")
root.configure(bg="#2c3e50")

# Label to display image/video output
label = tk.Label(root, bg="#34495e", relief="sunken", bd=2)
label.pack(pady=20, ipadx=5, ipady=5)

def process_image(image_path):
    """Processes an image and displays detection results."""
    img = cv2.imread(image_path)
    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if model.names[cls] in ["car", "truck", "bus", "motorcycle"]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label_text = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((700, 500))  
    img_tk = ImageTk.PhotoImage(img)

    label.config(image=img_tk)
    label.image = img_tk  

def process_video(video_path):
    """Processes a video file and displays detection results in a loop."""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 450))  
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if model.names[cls] in ["car", "truck", "bus", "motorcycle"]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Car Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_image():
    """Opens file dialog to select and process an image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path)

def upload_video():
    """Opens file dialog to select and process a video."""
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        threading.Thread(target=process_video, args=(file_path,)).start()

btn_image = tk.Button(root, text="📸 Upload Image", command=upload_image, font=("Arial", 14), bg="#27ae60", fg="white", width=20, height=2)
btn_image.pack(pady=10)

btn_video = tk.Button(root, text="🎥 Upload Video", command=upload_video, font=("Arial", 14), bg="#2980b9", fg="white", width=20, height=2)
btn_video.pack(pady=10)

root.mainloop()
