# ASL Fingerspelling Recognition and Real-Time Translator

A computer vision and machine learning project for recognizing **American Sign Language (ASL) fingerspelling gestures** from a webcam and converting detected hand gestures into letters and text in real time.

The project uses **MediaPipe Hands** to extract 21 hand landmarks from a detected hand. Each landmark contains `x`, `y`, and `z` coordinates, producing **63 numerical features per sample**. These features are then processed and classified using multiple Support Vector Machine (SVM) models.

The final application extends gesture recognition into a simple **real-time fingerspelling translator**, where a detected gesture is accepted as a letter when the user releases the hand, allowing letters to be accumulated into words and sentences.

> **Current scope:** This project focuses primarily on static ASL fingerspelling gestures. Dynamic signs whose meaning depends on movement over time are not fully modeled by the current single-frame SVM pipeline and are a future extension using temporal models such as LSTM, GRU, or Transformers.

---

## Demo

The repository contains both model-evaluation results and the final real-time application.

### Model Evaluation

#### Evaluation Result 1

![Model Evaluation 1](Demo_Results/Evaluating%20the%20model.png)

#### Evaluation Result 2

![Model Evaluation 2](Demo_Results/Evaluating%20the%20model%202.png)

### Fingerspelling Application

![Fingerspelling Application](Demo_Results/FingerSpelling%20Application.png)

### Full Project Demonstration

[Watch the complete ASL Fingerspelling Translator demonstration](Demo_Results/ASL-Fingerspelling-Translator.mp4)

---

# 1. Project Overview

The project follows an end-to-end computer vision and machine learning pipeline:

```text
ASL Image Dataset
        │
        ▼
Landmark Extraction
        │
        ▼
MediaPipe Hands
        │
        ▼
21 Hand Landmarks
        │
        ▼
21 × 3 coordinates
        │
        ▼
63 Numerical Features
        │
        ▼
Preprocessing
        │
        ├── Raw Landmarks
        │
        └── Wrist-Centered Landmarks
        │
        ▼
StandardScaler
        │
        ▼
SVM Training
        │
        ├── Raw Linear SVM
        ├── Wrist-Centered Linear SVM
        └── RBF SVM
        │
        ▼
Validation and Evaluation
        │
        ▼
Real-Time Webcam Prediction
        │
        ▼
Prediction Stabilization
        │
        ▼
Gesture Acceptance
        │
        ▼
Letter
        │
        ▼
Fingerspelling Text
```

---

# 2. Main Features

- MediaPipe-based hand landmark extraction
- 21 hand landmarks per detected hand
- 63 numerical features per sample
- Raw landmark representation
- Wrist-centered landmark normalization
- StandardScaler preprocessing
- Linear SVM classification
- RBF SVM classification
- Validation accuracy evaluation
- Precision, recall, and F1-score evaluation
- Confusion matrices
- Real-time webcam inference
- Real-time model switching
- Prediction confidence display
- Majority-vote prediction stabilization
- Gesture-release based letter acceptance
- Letter accumulation into words and sentences
- `space` and `del` gesture support
- Mirrored webcam visualization
- Jupyter Notebook-based workflow
- Saved trained models for reuse

---

# 3. Dataset

The project is based on the following Kaggle dataset:

**ASL (American Sign Language) Alphabet Dataset**

Dataset:
https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset

The original dataset contains ASL alphabet gesture images.

Instead of training directly on RGB images, the project processes the images using MediaPipe Hands and converts each detected hand into a numerical landmark representation.

Each detected hand contains:

```text
21 landmarks
×
3 coordinates (x, y, z)
=
63 numerical features
```

The resulting landmark data follows the general structure:

```text
label | x0 | y0 | z0 | x1 | y1 | z1 | ... | x20 | y20 | z20
```

## Dataset Classes

The project uses the following class mapping:

```text
A     -> 0
B     -> 1
C     -> 2
D     -> 3
E     -> 4
F     -> 5
G     -> 6
H     -> 7
I     -> 8
J     -> 9
K     -> 10
L     -> 11
M     -> 12
N     -> 13
O     -> 14
P     -> 15
Q     -> 16
R     -> 17
S     -> 18
T     -> 19
U     -> 20
V     -> 21
W     -> 22
X     -> 23
Y     -> 24
Z     -> 25
del   -> 26
space -> 27
```

> The original image dataset is not included in this repository. The repository contains processed landmark data derived from the dataset.

---

# 4. Processed Dataset

The processed data is included in the repository under:

```text
DataSet/
└── processed/
    ├── landmarks.csv
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

The processed data contains numerical landmark features extracted using MediaPipe.

This allows users to reproduce the training and evaluation stages without requiring the original image dataset to be stored in the repository.

Because the processed CSV files are large, they are managed using **Git LFS**.

If Git LFS is not already installed on your machine:

```bash
git lfs install
```

After cloning the repository, make sure the LFS files are downloaded:

```bash
git lfs pull
```

---

# 5. Repository Structure

```text
ASL-Fingerpelling-Translator-Project/
│
├── DataSet/
│   └── processed/
│       ├── landmarks.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── Demo&Results/
│   ├── ASL-Fingerspelling-Translator.mp4
│   ├── Evaluating the model.png
│   ├── Evaluating the model 2.png
│   └── FingerSpelling Application.png
│
├── models/
│   ├── linear_svm_raw.pkl
│   ├── raw_scaler.pkl
│   ├── rbf_svc.pkl
│   └── raw_linear_svm.pkl
│
├── src/
│   ├── config.py
│   ├── Preprocess.ipynb
│   ├── extract_landmarks.ipynb
│   ├── train_model.ipynb
│   ├── evaluate_models.ipynb
│   ├── realtime_prediction_test.ipynb
│   └── fingerspelling_translator.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 6. Technologies Used

## Computer Vision

- OpenCV
- MediaPipe Hands

## Machine Learning

- Scikit-learn
- Support Vector Machines
- Linear SVM
- RBF SVM

## Data Processing

- NumPy
- Pandas
- StandardScaler

## Evaluation

- Scikit-learn metrics
- Matplotlib
- Seaborn
- Confusion matrices
- Classification reports

## Model Serialization

- Joblib

## Development

- Python
- Jupyter Notebook

---

# 7. Requirements

The project was developed and tested using:

```text
Python 3.11.5
```

The main dependencies are:

```text
numpy==1.26.4
pandas
scikit-learn
opencv-python
mediapipe
matplotlib
seaborn
joblib
jupyter
ipykernel
```

All required packages are listed in:

```text
requirements.txt
```

---

# 8. Installation

Clone the repository:

```bash
git clone https://github.com/shehababdo/ASL-Fingerspelling-Translator.git
```

Move into the project directory:

```bash
cd ASL-Fingerspelling-Translator
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If Git LFS is required:

```bash
git lfs install
git lfs pull
```

Start Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Then open the notebooks inside:

```text
src/
```

---

# 9. Complete Workflow

The recommended workflow is:

```text
1. Extract landmarks
        ↓
2. Preprocess the data
        ↓
3. Train the models
        ↓
4. Evaluate the models
        ↓
5. Test real-time recognition
        ↓
6. Run the fingerspelling translator
```

The corresponding notebooks are:

```text
src/extract_landmarks.ipynb
        ↓
src/Preprocess.ipynb
        ↓
src/train_model.ipynb
        ↓
src/evaluate_models.ipynb
        ↓
src/realtime_prediction_test.ipynb
        ↓
src/fingerspelling_translator.ipynb
```

If the processed dataset and saved models are already present, the extraction and training stages can be skipped.

---

# 10. Step 1 — Extract Hand Landmarks

Open:

```text
src/extract_landmarks.ipynb
```

This notebook processes the original ASL gesture images using MediaPipe Hands.

For each image:

```text
Image
  ↓
MediaPipe Hands
  ↓
Hand Detection
  ↓
21 Landmarks
  ↓
x, y, z Coordinates
  ↓
63 Features
```

The extracted landmark dataset is saved into the processed-data directory.

Images for which a hand cannot be detected may be skipped.

---

# 11. Step 2 — Preprocess the Landmark Data

Open:

```text
src/Preprocess.ipynb
```

The preprocessing stage separates labels and features and prepares different landmark representations for machine learning.

## Raw Landmarks

The original MediaPipe coordinates are used directly.

```text
Raw Landmarks
      ↓
StandardScaler
      ↓
Linear SVM
```

## Wrist-Centered Landmarks

The landmarks are normalized relative to the wrist landmark.

The wrist is landmark `0`.

The transformation is:

```python
def wrist_center(X):

    X = X.reshape(-1, 21, 3)

    wrist = X[:, 0, :]

    X = X - wrist[:, np.newaxis, :]

    return X.reshape(-1, 63)
```

This changes the representation from absolute hand position to landmark positions relative to the wrist.

The feature shape remains:

```text
(number_of_samples, 63)
```

The wrist-centered representation produced a significant improvement in the current experiments.

---

# 12. Step 3 — Train the Models

Open:

```text
src/train_model.ipynb
```

Three main SVM configurations were evaluated.

## Model 1 — Raw Linear SVM

```text
Raw Landmarks
      ↓
StandardScaler
      ↓
Linear SVM
```

## Model 2 — Wrist-Centered Linear SVM

```text
Raw Landmarks
      ↓
Wrist-Centered Normalization
      ↓
StandardScaler
      ↓
Linear SVM
```

## Model 3 — RBF SVM

```text
Processed Landmarks
      ↓
StandardScaler
      ↓
RBF SVM
```

A typical Linear SVM configuration used:

```python
from sklearn.svm import SVC

linear_svm = SVC(
    kernel="linear",
    C=1,
    probability=True
)
```

The RBF model uses:

```python
kernel="rbf"
```

The `probability=True` option enables probability estimates through `predict_proba()` for the real-time application.

---

# 13. SVM Parameters

## Kernel

### Linear

```python
kernel="linear"
```

Creates a linear decision boundary.

### RBF

```python
kernel="rbf"
```

Allows nonlinear decision boundaries.

## C

```python
C=1
```

Controls the trade-off between the complexity of the decision boundary and classification errors.

## Probability

```python
probability=True
```

Enables:

```python
model.predict_proba(X)
```

which is used by the real-time application to display an estimated prediction confidence.

---

# 14. Saved Models and Scalers

The trained models and scalers are serialized using Joblib.

Example:

```python
import joblib

joblib.dump(
    linear_svm,
    "../models/linear_svm_raw.pkl"
)

joblib.dump(
    raw_scaler,
    "../models/raw_scaler.pkl"
)

joblib.dump(
    rbf_svm,
    "../models/rbf_svc.pkl"
)

joblib.dump(
    raw_linear_svm,
    "../models/raw_linear_svm.pkl"
)
```

A scaler is part of the preprocessing pipeline and must match the preprocessing used during training.

---

# 15. Important: Scaler vs Model

A scaler is **not** a classifier.

For example:

```python
raw_scaler
```

is a preprocessing object.

While:

```python
linear_svm
```

is the trained classifier.

The inference pipeline must therefore be:

```text
Landmarks
    ↓
Correct preprocessing
    ↓
Correct scaler
    ↓
Correct SVM
    ↓
Prediction
```

The preprocessing used for inference must be the same preprocessing used when training the model.

---

# 16. Step 4 — Evaluate the Models

Open:

```text
src/evaluate_models.ipynb
```

The validation dataset is used to compare the models during development.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Classification reports
- Confusion matrices

Example:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_val,
    y_val_pred
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)
```

Accuracy measures the percentage of validation samples that were classified correctly.

However, overall accuracy alone does not explain which individual classes are difficult.

For this reason, the classification report and confusion matrix are also used.

---

# 17. Validation Results

The current experiments produced:

| Model | Validation Accuracy |
|---|---:|
| Raw Linear SVM | **72.29%** |
| Wrist-Centered Linear SVM | **98.48%** |
| RBF SVM | **98.45%** |

The wrist-centered Linear SVM achieved the best overall validation accuracy.

The RBF SVM achieved an almost identical overall result but showed different behavior for individual classes.

The raw-landmark Linear SVM performed substantially worse.

The results demonstrate that **feature representation and normalization can have a major impact on classification performance**.

---

# 18. Real-Time Model Testing

Open:

```text
src/realtime_prediction_test.ipynb
```

The real-time pipeline is:

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hands
   ↓
21 Landmarks
   ↓
63 Features
   ↓
Scaler
   ↓
SVM
   ↓
Prediction
```

The image used for machine-learning inference remains in its original orientation.

The frame is flipped only for visualization:

```python
display_frame = cv2.flip(frame, 1)
```

This creates a natural mirror-like webcam experience while keeping the ML input consistent.

---

# 19. Real-Time Model Switching

The real-time testing notebook supports model switching while the webcam is running.

Controls:

```text
1 → Linear SVM
2 → RBF SVM
3 → Wrist-Centered Linear SVM
Q → Quit
```

This makes it possible to compare different models using the same webcam input.

Prediction history is cleared whenever the model is switched.

---

# 20. Prediction Stabilization

Individual webcam frames can occasionally be misclassified.

To reduce frame-to-frame instability, predictions are stored in a short history:

```python
from collections import deque, Counter

prediction_history = deque(maxlen=10)
```

A majority vote is then used:

```python
stable_label = Counter(
    prediction_history
).most_common(1)[0][0]
```

For example:

```text
Frame 1 → A
Frame 2 → A
Frame 3 → B
Frame 4 → A
Frame 5 → A
```

can result in:

```text
Stable prediction → A
```

This makes the real-time output more stable.

---

# 21. Fingerspelling Translator

Open:

```text
src/fingerspelling_translator.ipynb
```

The fingerspelling translator builds on the real-time recognition system.

Instead of only showing the current predicted letter, the application maintains a text buffer.

The process is:

```text
Hand Gesture
      ↓
SVM Prediction
      ↓
Prediction Stabilization
      ↓
Stable Gesture
      ↓
Hand Release / Gesture Completion
      ↓
Accept Letter
      ↓
Add Letter to Text
```

For example:

```text
Show H
Release hand
→ H

Show E
Release hand
→ HE

Show L
Release hand
→ HEL

Show L
Release hand
→ HELL

Show O
Release hand
→ HELLO
```

Repeated letters are also supported:

```text
L → release → L → release
```

results in:

```text
LL
```

---

# 22. Special Gestures

The classifier also contains:

```text
space
del
```

## Space

Performing the `space` gesture and releasing the hand inserts a space:

```text
HELLO
+
space
=
HELLO WORLD
```

## Delete

Performing the `del` gesture and releasing the hand removes the last character.

For example:

```text
HELLO
+
del
=
HELL
```

---

# 23. Real-Time Visualization

The real-time application displays:

- Current prediction
- Estimated confidence
- Current model
- Accumulated text
- Keyboard controls
- Hand landmarks

The display is mirrored for easier interaction while inference is performed on the original frame orientation.

---

# 24. Current Observations

Validation results and real-time behavior do not always match exactly.

The current experiments showed that the wrist-centered Linear SVM performs very well for several static signs, while some letters remain challenging during webcam inference.

Real-time testing also showed that visually similar gestures can still produce confusion even when overall validation accuracy is very high.

This is why both:

1. Offline evaluation
2. Real-world webcam testing

are necessary when developing a real-time computer-vision system.

---

# 25. Limitations

## Static Gesture Assumption

The SVM models process one frame at a time.

Therefore, they do not explicitly learn how hand landmarks move over time.

## Dynamic Gestures

Some signs can depend on movement and temporal information.

A single-frame classifier cannot fully represent that temporal structure.

A natural future direction is:

```text
Sequence of Landmarks
        ↓
LSTM / GRU / Transformer
        ↓
Dynamic Gesture Recognition
```

## Real-World Conditions

Performance can vary depending on:

- Lighting
- Camera quality
- Background
- Hand orientation
- Distance from camera
- Partial occlusion
- Individual signing style

## Dataset Bias

High validation accuracy on a dataset does not necessarily mean identical accuracy for every user in a real-world environment.

---

# 26. Future Improvements

Possible future improvements include:

- LSTM-based dynamic gesture recognition
- GRU-based temporal modeling
- Transformer-based gesture recognition
- Two-hand ASL recognition
- Improved gesture segmentation
- More robust gesture acceptance logic
- Better handling of repeated gestures
- Word-level recognition
- Sentence-level recognition
- Language-model-based sentence correction
- Text-to-speech output
- Mobile deployment
- Edge deployment
- Larger and more diverse datasets
- Improved handling of difficult gestures such as J and Z
- Evaluation with multiple users
- Better real-world robustness

---

# 27. Why Use Hand Landmarks?

Instead of training directly on RGB images, the project converts each hand into a compact numerical representation.

Rather than processing an entire image, the classifier receives:

```text
21 landmarks × 3 coordinates
=
63 numerical features
```

This provides a compact representation that is suitable for:

- Classical machine learning
- Fast inference
- Real-time applications
- Easier debugging
- Feature normalization
- Visualization
- Model comparison

---

# 28. Common Problems

## MediaPipe / NumPy Compatibility

The project was developed with NumPy 1.26.4.

If MediaPipe has compatibility problems with a newer NumPy version, install:

```bash
pip install numpy==1.26.4
```

Then restart the Jupyter kernel.

---

## OpenCV Camera Error

If:

```python
success, frame = cap.read()
```

returns `False`, check:

- Camera permissions
- Whether another program is using the webcam
- The camera index

For example:

```python
cap = cv2.VideoCapture(0)
```

If necessary:

```python
cap = cv2.VideoCapture(1)
```

---

## `cvtColor` Error

If OpenCV reports:

```text
(-215:Assertion failed) !_src.empty()
```

it usually means OpenCV received an empty frame.

Always check:

```python
success, frame = cap.read()

if not success:
    break
```

before calling:

```python
cv2.cvtColor(...)
```

---

## Wrong Predictions After Changing Preprocessing

The scaler and preprocessing used during inference must correspond to the model used during training.

Do not mix different preprocessing pipelines.

Correct:

```text
Training
    ↓
Wrist-Centered
    ↓
Scaler
    ↓
SVM

Inference
    ↓
Wrist-Centered
    ↓
Same Scaler
    ↓
Same SVM
```

---

# 29. Reproducing the Project

For a complete reproduction:

```text
1. Clone the repository
        ↓
2. Install dependencies
        ↓
3. Install/configure Git LFS
        ↓
4. Download the original ASL image dataset from Kaggle if required
        ↓
5. Configure paths
        ↓
6. Extract landmarks
        ↓
7. Preprocess the data
        ↓
8. Train the models
        ↓
9. Evaluate the models
        ↓
10. Run real-time prediction
        ↓
11. Run the fingerspelling translator
```

If the processed dataset and trained models included in the repository are used, the extraction and training stages can be skipped.

---

# 30. Quick Start

Clone the repository:

```bash
git clone https://github.com/shehababdo/ASL-Fingerspelling-Translator.git
```

Enter the project directory:

```bash
cd ASL-Fingerspelling-Translator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If Git LFS is required:

```bash
git lfs install
git lfs pull
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
src/realtime_prediction_test.ipynb
```

for real-time model testing, or:

```text
src/fingerspelling_translator.ipynb
```

for the complete fingerspelling application.

---

# 31. Project Results

The current experiments demonstrate that landmark representation has a significant effect on machine-learning performance.

The strongest current pipeline was:

```text
MediaPipe Hands
       ↓
Wrist-Centered Normalization
       ↓
StandardScaler
       ↓
Linear SVM
```

with:

```text
98.48% validation accuracy
```

The RBF SVM achieved:

```text
98.45% validation accuracy
```

while the raw-landmark Linear SVM achieved:

```text
72.29% validation accuracy
```

This demonstrates that selecting a suitable feature representation can be more important than simply increasing classifier complexity.

---

# 32. Dataset Reference

The original dataset used in this project:

**ASL (American Sign Language) Alphabet Dataset — Kaggle**

https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset

Please review the dataset's original license and usage terms before redistributing or using derived data outside this repository.

---

# 33. Author

## Shehab Abdo

Mechatronics & Robotics Engineer

**Portfolio:**  
https://shehababdo.github.io/

**LinkedIn:**  
https://www.linkedin.com/in/shehab-abdo-a94946198/

---

# 34. Acknowledgements

This project uses:

- **MediaPipe Hands** for hand landmark detection
- **OpenCV** for computer vision and webcam processing
- **Scikit-learn** for machine-learning models
- **NumPy** for numerical processing
- **Pandas** for data processing
- **Matplotlib** for visualization
- **Seaborn** for evaluation visualization
- **Joblib** for model serialization

---

# 35. License

This repository can be distributed under an open-source license such as the MIT License, provided that the repository's license is compatible with the licenses and terms of the dataset and third-party components used by the project.

Check the original dataset license before redistributing the dataset or derived materials beyond what its terms permit.

---

# 36. Final Project Summary

This project represents an end-to-end implementation of a real-time ASL fingerspelling recognition pipeline:

```text
ASL Dataset
     ↓
MediaPipe Hand Landmarks
     ↓
Feature Engineering
     ↓
Wrist-Centered Normalization
     ↓
StandardScaler
     ↓
SVM Classification
     ↓
Model Evaluation
     ↓
Real-Time Webcam Inference
     ↓
Prediction Stabilization
     ↓
Gesture Acceptance
     ↓
Fingerspelling
     ↓
Text
```

The project demonstrates how a lightweight landmark-based representation combined with classical machine learning can be used to build a practical real-time computer vision system.

The next major direction is extending the system from static fingerspelling recognition toward temporal modeling for dynamic signs and continuous sign-language understanding.
