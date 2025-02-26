# Object Detection and Image Matching with YOLO

This project provides a Python-based solution for detecting objects (specifically dogs or persons) in images using the YOLO (You Only Look Once) object detection model and finding similar images within a database directory, including its subdirectories.

## Project Overview

The program takes a query image and a database directory as input. It then uses YOLO to detect objects of a specified class (dog or person) in the query image. Subsequently, it searches through the database directory (and its subdirectories recursively) to find images that contain objects of the same class. Matching images are then copied to a separate output directory.

## Requirements

To run this project, you'll need the following:

* **Python 3.6+:** Python 3.6 or a later version is required.
* **Libraries:**
    * `opencv-python`: For image processing and YOLO inference. Install using: `pip install opencv-python`
    * `numpy`: For numerical operations. Install using: `pip install numpy`
    * `tqdm`: For progress bars. Install using: `pip install tqdm`
    * `argparse`: For command-line argument parsing (included in Python standard library).
* **YOLOv3 Model Files:**
    * `yolov3.weights`: The pre-trained YOLOv3 weights file.
    * `yolov3.cfg`: The YOLOv3 configuration file.
    * `coco.names`: The file containing class names that YOLO was trained on.
    * These files can be downloaded from reputable sources online, such as the official YOLO website or GitHub repositories.
* **Configuration File:**
    * `config.json`: A JSON file containing the paths to the query image and database directory.

## How to Replicate

1.  **Set Up a Virtual Environment (Recommended):**
    * Create a virtual environment to isolate project dependencies:
        ```bash
        python -m venv myenv
        myenv\Scripts\activate  # Windows
        source myenv/bin/activate  # macOS/Linux
        ```
2.  **Install Dependencies:**
    * Install the required Python libraries using pip:
        ```bash
        pip install opencv-python numpy tqdm
        ```
3.  **Download YOLO Files:**
    * Download `yolov3.weights`, `yolov3.cfg`, and `coco.names` from a trusted source.
    * Place these files in the same directory as your Python script.
4.  **Create `config.json`:**
    * Create a `config.json` file in the same directory as your Python script with the following structure:
        ```json
        {
          "query_image_path": "C:\\path\\to\\your\\query_image.jpg",
          "database_dir": "C:\\path\\to\\your\\image\\database\\directory"
        }
        ```
    * Replace the example paths with the actual paths to your query image and database directory.
5.  **Run the Script:**
    * Open your terminal or command prompt.
    * Navigate to the directory containing your Python script.
    * Run the script with the desired target class (person or dog) as a command-line argument:
        ```bash
        python your_script_name.py person  # To find images with persons
        python your_script_name.py dog     # To find images with dogs
        ```
    * Replace `your_script_name.py` with the actual name of your Python file.
6.  **Output:**
    * Matching images will be copied to a newly created directory named `Matching_<target_class>s_<timestamp>`.

## Code Description

* **`detect_objects_yolo(image_path, net, classes, target_class, conf_threshold, nms_threshold)`:**
    * Detects objects in an image using YOLO, filtering for the specified `target_class`.
* **`find_matching_objects_yolo(query_image_path, database_dir, net, classes, target_class)`:**
    * Finds images in the `database_dir` (and its subdirectories) that contain objects matching those detected in the `query_image`.
* **`copy_matching_images(matching_images, output_dir)`:**
    * Copies the found matching images to the specified `output_dir`.
* **`main()`:**
    * Handles command-line arguments, loads the YOLO model, reads paths from `config.json`, performs object detection and matching, and copies the results.
    * Includes error handling for file not found, incorrect json format, and incorrect command line arguments.
    * Recursively searches subfolders within the database folder.

## Notes

* Ensure that the YOLO model files are compatible with the classes you intend to detect.
* The accuracy of object detection depends on the quality of the YOLO model and the images.
* The script recursively searches all subdirectories within the specified database directory.
* The confidence and NMS thresholds can be adjusted to fine-tune the detection results.