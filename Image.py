import cv2
from ultralytics import YOLO
import torch

# Check if MPS (Apple's GPU) is available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

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

# Function to process and predict on an image
def predict_image(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Perform inference
    results = model(image, conf=0.25)

    # Annotate the image
    for box in results[0].boxes:
        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        label_index = int(box.cls[0])  # Class label index

        # Get the label text
        label_text = f"{class_names[label_index]} ({confidence:.2f})"

        # Determine the color based on the label
        category = class_names[label_index]
        box_color = colors.get(category, (255, 255, 255))  # Default to white if no match

        # Draw bounding box and label on the image
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        # Ensure the text is not placed outside the image
        y_text = max(y1 - 10, 10)  # Clamp the y-coordinate to avoid negative values

        # Get the text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Define the background rectangle coordinates
        background_start = (x1, y_text - text_height - 4)
        background_end = (x1 + text_width, y_text + baseline - 4)

        # Draw the background rectangle
        cv2.rectangle(image, background_start, background_end, (0, 0, 0), -1)  # Black background

        # Draw the label text in white
        cv2.putText(image, label_text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the annotated image
    cv2.imshow("YOLOv8 Mask Detection - Image", image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the window

# Example usage for image prediction
image_path = "/Users/deepanshusehgal/Developer/Code/AI_ML/apple-silicon/YOLO_dataset/train/images/000001_3_000001_MALE_25.jpg"  # Replace with the path to your image
predict_image(image_path, model)
