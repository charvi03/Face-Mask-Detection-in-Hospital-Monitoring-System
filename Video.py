import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('/Users/deepanshusehgal/Developer/Code/AI_ML/apple-silicon/mask_detection_project/yolov8_mask_detection12/weights/best.pt')

# Define class names and corresponding colors
class_names = ['mask_worn_correctly', 'no_mask', 'mask_not_on_nose', 'mask_on_chin']
colors = {
    'mask_worn_correctly': (0, 255, 0),  # Green
    'no_mask': (0, 0, 255),              # Red
    'mask_not_on_nose': (0, 255, 255),   # Yellow
    'mask_on_chin': (0, 255, 255)        # Yellow
}

# Path to the video file
video_path = "/Users/deepanshusehgal/Developer/Code/AI_ML/apple-silicon/app/video.mp4"  # Replace with your video file path
output_path = "/Users/deepanshusehgal/Developer/Code/AI_ML/apple-silicon/app/annotated_video.mp4"  # Output file path

cap = cv2.VideoCapture(video_path)  # Open the video file

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter for saving the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Loop through video frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Crop the face region
        face_roi = frame[y:y + h, x:x + w]

        # Resize the cropped face for YOLO inference
        face_resized = cv2.resize(face_roi, (416, 416))

        # Perform mask detection on the face
        results = model(face_resized, conf=0.25, half=False)

        # Get the best detection result
        for box in results[0].boxes:
            label_index = int(box.cls[0])  # Get label index
            confidence = box.conf[0]  # Confidence score
            category = class_names[label_index]
            box_color = colors.get(category, (255, 255, 255))  # Default to white

            # Draw bounding box and label on the original frame
            label_text = f"{category} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the annotated frame
    cv2.imshow("Real-Time Face Mask Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to: {output_path}")
