import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2

# Define the categories and their corresponding class IDs
class_mapping = {
    'mask_worn_correctly': 0,
    'no_mask': 1,
    'mask_not_on_nose': 2,
    'mask_on_chin': 3
}

# YOLO formatted dataset output paths
train_images_dir = 'YOLO_dataset/train/images'
train_labels_dir = 'YOLO_dataset/train/labels'
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)

# Function to convert bounding boxes to YOLO format
def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Preprocess a single new image
def preprocess_new_image(image_path, category_label):
    # Validate the category label
    if category_label not in class_mapping:
        raise ValueError(f"Invalid category label: {category_label}")

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    img_height, img_width, _ = img.shape

    # Resize the image while maintaining the aspect ratio
    target_size = 640  # Target resolution
    scale = target_size / max(img_width, img_height)
    resized_width = int(img_width * scale)
    resized_height = int(img_height * scale)

    # Resize with a high-quality interpolation method
    resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

    # Pad the resized image to make it square
    padded_img = cv2.copyMakeBorder(
        resized_img,
        top=(target_size - resized_height) // 2,
        bottom=(target_size - resized_height) - (target_size - resized_height) // 2,
        left=(target_size - resized_width) // 2,
        right=(target_size - resized_width) - (target_size - resized_width) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )

    # Create bounding box (entire image as bounding box)
    face_bbox = [0, 0, img_width, img_height]
    yolo_bbox = convert_bbox_to_yolo_format(face_bbox, target_size, target_size)

    # Generate a unique name for the image
    base_name = os.path.basename(image_path)
    new_image_name = os.path.splitext(base_name)[0] + '_processed.jpg'

    # Save the preprocessed image
    output_img_path = os.path.join(train_images_dir, new_image_name)
    cv2.imwrite(output_img_path, padded_img)

    # Create annotation file
    txt_filename = os.path.splitext(new_image_name)[0] + '.txt'
    txt_filepath = os.path.join(train_labels_dir, txt_filename)

    with open(txt_filepath, 'w') as f:
        f.write(f"{class_mapping[category_label]} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

    print(f"Preprocessed image saved: {output_img_path}")
    print(f"Annotation file created: {txt_filepath}")

# File watcher class
class NewImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Trigger when a new file is created
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"New image detected: {event.src_path}")
            # Preprocess the new image with a default category (update as needed)
            category_label = 'mask_worn_correctly'  # Replace with logic to determine category
            preprocess_new_image(event.src_path, category_label)

# Monitor a directory for new images
def start_monitoring(directory_to_watch):
    print(f"Monitoring directory: {directory_to_watch}")
    event_handler = NewImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Start monitoring a directory (e.g., `new_images/`)
new_images_dir = 'new_images'
os.makedirs(new_images_dir, exist_ok=True)
start_monitoring(new_images_dir)
