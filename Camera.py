import cv2
from ultralytics import YOLO
import torch
import pygame
import threading

# Check if MPS (Apple's GPU) is available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load your trained YOLOv8 model
model = YOLO('/Users/deepanshusehgal/Developer/Code/AI_ML/apple-silicon/mask_detection_project/yolov8_mask_detection12/weights/best.pt')

# Initialize pygame mixer for audio alerts
pygame.mixer.init()
beep_sound_path = "/Users/deepanshusehgal/Developer/Code/AI_ML/apple-silicon/app/beep.mp3"  # Replace with the full path to your beep sound file
pygame.mixer.music.load(beep_sound_path)

# Function to play the alert sound (non-blocking)
def play_alert_sound():
    if not pygame.mixer.music.get_busy():  # Prevent overlapping sounds
        threading.Thread(target=pygame.mixer.music.play, daemon=True).start()

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Frame skipping for better performance
frame_skip = 3
frame_count = 0

# Define class names and corresponding colors
class_names = ['mask_worn_correctly', 'no_mask', 'mask_not_on_nose', 'mask_on_chin']
colors = {
    'mask_worn_correctly': (0, 255, 0),  # Green
    'no_mask': (0, 0, 255),              # Red
    'mask_not_on_nose': (0, 255, 255),   # Yellow
    'mask_on_chin': (0, 255, 255)        # Yellow
}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    # Skip frames to reduce workload
    if frame_count % frame_skip != 0:
        continue

    # Ensure frame capture is successful
    if frame is not None:
        # Resize the frame for faster processing
        frame = cv2.resize(frame, (416, 416))  # Adjust size as needed for speed vs accuracy

        # Perform inference
        results = model(frame, conf=0.25, half=False)  # `half` is not used with MPS

        # Retrieve detections and plot annotations on the frame
        annotated_frame = results[0].plot()

        # Loop through detections
        for box in results[0].boxes:
            label_index = int(box.cls[0])  # Get label index
            confidence = box.conf[0]  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Get the label text
            label_text = f"{class_names[label_index]} ({confidence:.2f})"

            # Determine the color based on the label
            category = class_names[label_index]
            box_color = colors.get(category, (255, 255, 255))  # Default to white if no match

            # Play alert sound for specific labels
            if label_index in [1, 2, 3]:  # 'no_mask', 'mask_not_on_nose', 'mask_on_chin'
                play_alert_sound()

            # Draw bounding box and label on the annotated frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Display the frame with detections
        cv2.imshow("YOLOv8 Real-Time Mask Detection", annotated_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
