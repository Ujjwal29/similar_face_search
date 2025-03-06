import os
import shutil
import time
import cv2
import numpy as np
from tqdm import tqdm
import json

def detect_dog_faces_yolo(image_path, net, classes, conf_threshold=0.5, nms_threshold=0.4):
    """Detects dog faces using YOLO."""
    try:
        img = cv2.imread(image_path)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold and classes[class_id] == 'dog':
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        dog_faces = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                dog_faces.append((x, y, w, h))

        return dog_faces

    except Exception as e:
        print(f"Error detecting dog faces in {image_path}: {e}")
        return []

def find_matching_dog_faces_yolo(query_image_path, database_dir, net, classes):
    """Finds images with matching dog faces using YOLO, recursively searching subdirectories."""
    query_faces = detect_dog_faces_yolo(query_image_path, net, classes)
    if not query_faces:
        print("No dog faces found in the query image.")
        return []

    matching_images = []

    def search_directory(directory):
        """Recursively searches a directory for image files."""
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_path = os.path.join(root, filename)
                    database_faces = detect_dog_faces_yolo(file_path, net, classes)

                    if database_faces:
                        matching_images.append(file_path)

    search_directory(database_dir) #Start recursive search.

    return matching_images

def copy_matching_images(matching_images, output_dir):
    """Copies images with matching dog faces to a new directory."""
    os.makedirs(output_dir, exist_ok=True)
    for matching_image_path in tqdm(matching_images, desc="Copying matching images"):
        try:
            shutil.copy2(matching_image_path, output_dir)
        except Exception as e:
            print(f"Error copying {matching_image_path}: {e}")

def main():
    """Main function to execute dog face matching and copying using YOLO, recursively searching subdirectories."""

    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Read paths from config.json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            query_image_path = config["query_image_path"]
            database_dir = config["database_dir"]
    except FileNotFoundError:
        print("Error: config.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in config.json.")
        return
    except Exception as e:
        print(f"Error reading paths from config.json: {e}")
        return

    # Check if the paths are valid.
    if not os.path.isfile(query_image_path):
        print(f"Error: Query image '{query_image_path}' not found.")
        return

    if not os.path.isdir(database_dir):
        print(f"Error: Database directory '{database_dir}' not found.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"Matching_Dog_Faces_{timestamp}"

    matching_images = find_matching_dog_faces_yolo(query_image_path, database_dir, net, classes)

    if matching_images:
        copy_matching_images(matching_images, output_dir)
        print(f"Found {len(matching_images)} images with matching dog faces. Copied to {output_dir}")
    else:
        print("No images with matching dog faces found.")

if __name__ == "__main__":
    main()