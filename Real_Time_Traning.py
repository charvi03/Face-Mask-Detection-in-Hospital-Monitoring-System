import os
import time
import yaml
import cv2
from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch

# Paths for temporary dataset and YAML file
temp_dataset_dir = 'temp_dataset'
temp_images_dir = os.path.join(temp_dataset_dir, 'images')
temp_labels_dir = os.path.join(temp_dataset_dir, 'labels')
os.makedirs(temp_images_dir, exist_ok=True)
os.makedirs(temp_labels_dir, exist_ok=True)
temp_yaml_path = os.path.join(temp_dataset_dir, 'temp_data.yaml')

# Define the categories and their corresponding class IDs
class_mapping = {
    'mask_worn_correctly': 0,
    'no_mask': 1,
    'mask_not_on_nose': 2,
    'mask_on_chin': 3
}

# Create the temporary YAML file
def create_temp_yaml():
    temp_yaml_content = {
        'train': temp_images_dir,
        'val': temp_images_dir,  # Use the same image for validation
        'nc': len(class_mapping),  # Number of classes
        'names': list(class_mapping.keys())  # Class names
    }
    with open(temp_yaml_path, 'w') as yaml_file:
        yaml.dump(temp_yaml_content, yaml_file)
    print(f"Temporary YAML file created at: {temp_yaml_path}")

# Function to preprocess the image and save it
def preprocess_image(image_path, category_label):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None, None

    img_height, img_width, _ = img.shape

    # Resize the image for YOLO (square image with padding)
    target_size = 640
    scale = target_size / max(img_width, img_height)
    resized_width = int(img_width * scale)
    resized_height = int(img_height * scale)

    resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    padded_img = cv2.copyMakeBorder(
        resized_img,
        top=(target_size - resized_height) // 2,
        bottom=(target_size - resized_height) - (target_size - resized_height) // 2,
        left=(target_size - resized_width) // 2,
        right=(target_size - resized_width) - (target_size - resized_width) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )

    # Save the preprocessed image
    base_name = os.path.basename(image_path)
    temp_image_name = os.path.splitext(base_name)[0] + '_processed.jpg'
    temp_image_path = os.path.join(temp_images_dir, temp_image_name)
    cv2.imwrite(temp_image_path, padded_img)

    # Create YOLO annotation (entire image as bounding box)
    yolo_bbox = [0.5, 0.5, 1.0, 1.0]  # Centered, full width and height
    txt_filename = os.path.splitext(temp_image_name)[0] + '.txt'
    temp_label_path = os.path.join(temp_labels_dir, txt_filename)

    with open(temp_label_path, 'w') as f:
        f.write(f"{class_mapping[category_label]} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

    return temp_image_path, temp_label_path

# Classify the category of a new image using YOLO
def classify_image(image_path, model):
    results = model.predict(image_path, save=False, conf=0.25)
    if not results or not results[0].boxes:
        print(f"Could not classify the image: {image_path}")
        return None
    class_id = int(results[0].boxes.cls[0])
    category_label = [key for key, value in class_mapping.items() if value == class_id][0]
    print(f"Image classified as: {category_label}")
    return category_label

# Preprocess and train on a single image
def train_on_single_image(image_path, model):
    # Step 1: Classify the category
    category_label = classify_image(image_path, model)
    if category_label is None:
        print("Skipping training as classification failed.")
        return

    # Step 2: Preprocess the image and save it
    temp_image_path, temp_label_path = preprocess_image(image_path, category_label)
    if temp_image_path is None or temp_label_path is None:
        print("Skipping training as preprocessing failed.")
        return

    # Step 3: Create or update YAML file
    create_temp_yaml()

    # Step 4: Fine-tune the YOLO model with the single image
    print("Fine-tuning the model with the new image...")
    model.train(
        data=temp_yaml_path,
        epochs=1,  # Train for one epoch
        imgsz=640,
        batch=1,
        project='real_time_training',
        name='yolov8_mask_detection_single',
        device='mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print("Model fine-tuned successfully.")

    # Cleanup
    os.remove(temp_image_path)
    os.remove(temp_label_path)
    print("Temporary dataset cleaned up.")

# File watcher class to detect new images
class NewImageHandler(FileSystemEventHandler):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"New image detected: {event.src_path}")
            train_on_single_image(event.src_path, self.model)

# Monitor a directory for new images
def start_monitoring(directory_to_watch, model):
    print(f"Monitoring directory: {directory_to_watch}")
    event_handler = NewImageHandler(model)
    observer = Observer()
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Initialize YOLO model
yolo_model = YOLO('/Users/deepanshusehgal/Developer/Code/AI_ML/apple-silicon/mask_detection_project/yolov8_mask_detection12/weights/best.pt')

# Start monitoring a directory (e.g., `new_images/`)
new_images_dir = 'new_images'
os.makedirs(new_images_dir, exist_ok=True)
start_monitoring(new_images_dir, yolo_model)
