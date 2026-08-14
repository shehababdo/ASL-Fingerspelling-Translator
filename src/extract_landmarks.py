"""
extract_landmarks.py

This module extracts hand landmarks from the ASL image dataset using
MediaPipe Hands.

Workflow:
1. Load every image from the dataset.
2. Detect the hand using MediaPipe.
3. Extract the 21 hand landmarks.
4. Save the extracted landmark coordinates together with their class label.
5. Store the resulting dataset as a CSV file for machine learning.

This preprocessing step is executed only once and creates the dataset
used for all subsequent training and evaluation.

"""
"""
The pipeline
"""
"""
Initialize MediaPipe

↓

Open CSV file

↓

For each class

      ↓

For each image

      ↓

Load image

      ↓

Extract landmarks

      ↓

Write one CSV row

↓

Close CSV
"""

import csv
import config
import mediapipe as mp
import cv2
import numpy as np

def get_class_directories():
    class_directories = []

    for class_dir in config.TRAIN_DIR.iterdir():

        class_directories.append(class_dir)

    return class_directories

def collect_image_paths(class_directories):
    image_paths = []

    for class_dir in class_directories:

        label = class_dir.name

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

                image_paths.append((label, image_path))

    return image_paths

def preview_dataset(num_images=5):
    count = 0

    for class_dir in get_class_directories():

        label = class_dir.name

        for image_path in class_dir.iterdir():

            image = load_image(image_path)

            if image is None:
                continue

            print(f"Label: {label} | Image: {image_path.name}")

            cv2.imshow("Preview", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

            count += 1

            if count >= num_images:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

def initialize_mediapipe():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
    static_image_mode=config.STATIC_IMAGE_MODE,
    max_num_hands=config.MAX_NUM_HANDS,
    model_complexity=1,
    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )
    return hands

def load_image(image_path):
    """
    Load an image from disk and convert it to RGB.

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray | None
        RGB image if loading succeeds, otherwise None.
    """

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        return None

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb_img 

def extract_landmarks(image, hands):
    results = hands.process(image)

    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0].landmark # 0 because there is only one hand in the images, so 0 = first hand detected

    return None 

def flatten_landmarks(hand_landmarks):
    """
    Convert MediaPipe landmarks into a flat feature vector.

    Parameters
    ----------
    hand_landmarks : list
        List of 21 MediaPipe landmarks.

    Returns
    -------
    list[float]
        A list of 63 values:
        [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    """
    features = []

    for landmark in hand_landmarks:
        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)

    return features

def process_dataset(hands, writer):
    """
    Process the entire ASL dataset.

    For every image:
    1. Load the image.
    2. Detect the hand landmarks.
    3. Convert the landmarks into a feature vector.
    4. Write one row to the CSV file.
    """
    total_images = 0
    successful_images = 0

    for class_dir in get_class_directories():

        label = class_dir.name

        print(f"\nProcessing class: {label}")

        for image_path in class_dir.iterdir():

            # Skip non-image files
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            total_images += 1

            image = load_image(image_path)

            if image is None:
                continue

            landmarks = extract_landmarks(image, hands)

            if landmarks is None:
                continue

            features = flatten_landmarks(landmarks)

            save_csv(writer, label, features)

            successful_images += 1

            if total_images % 100 == 0:
                print(f"Processed {total_images} images...")

    print("\n========== Dataset Summary ==========")
    print(f"Total images      : {total_images}")
    print(f"Successful images : {successful_images}")
    print(f"Skipped images    : {total_images - successful_images}")


def save_csv(writer, label, features):
    """
    Write one processed image to the CSV file.

    Parameters
    ----------
    writer : csv.writer
        CSV writer object.

    label : str
        Image class (A, B, C, ...).

    features : list
        Flattened landmark vector (63 values).
    """

    row = [label] + list(features)

    writer.writerow(row)


#preview_dataset(10) Test the data Loader


def main():

    # Create output directory if it doesn't exist
    config.PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    hands = initialize_mediapipe()

    with open(config.LANDMARKS_CSV, "w", newline="") as file:

        writer = csv.writer(file)

        # Create CSV header
        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        writer.writerow(header)

        process_dataset(hands, writer)

    hands.close()

    print("\nCSV file created successfully!")
    print(config.LANDMARKS_CSV)

if __name__ == "__main__":
    main()