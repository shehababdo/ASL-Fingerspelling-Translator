# ASL Fingerspelling Recognition and Real-Time Translator

A computer-vision and machine-learning project for recognizing American Sign Language (ASL) hand gestures from a webcam and converting detected static gestures into letters and text in real time.

The project uses **MediaPipe Hands** to extract 21 hand landmarks (63 numerical features), followed by several **Support Vector Machine (SVM)** classifiers. The final real-time application can stabilize predictions, detect when the user releases a gesture, and build a written word/sentence on screen.

> **Current scope:** This project primarily focuses on static ASL fingerspelling gestures. Dynamic signs such as gestures whose meaning depends on motion over time are a separate problem and require temporal modeling such as LSTM/GRU/Transformer-based approaches.

---

## 1. Project Overview

The complete pipeline is:

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
21 landmarks × 3 coordinates
        │
        ▼
63 numerical features
        │
        ▼
Preprocessing
        │
        ├── Raw landmarks
        │
        └── Wrist-centered landmarks
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
Validation / Evaluation
        │
        ▼
Real-Time Webcam Prediction
        │
        ▼
Stable Gesture
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
- Raw landmark preprocessing
- Wrist-centered normalization
- StandardScaler preprocessing
- Linear SVM classification
- RBF SVM classification
- Validation accuracy and classification reports
- Confusion matrices
- Real-time webcam inference
- Real-time model switching
- Prediction confidence display
- Majority-vote prediction stabilization
- Letter accumulation into words/sentences
- `space` and `del` class support
- Visual mirrored webcam display
- Modular Jupyter Notebook workflow

---

# 3. Dataset

The original dataset consists of ASL gesture images.

The dataset was processed into landmark-based numerical data before model training. Instead of training directly on the RGB images, MediaPipe Hands was used to convert each image into hand landmark coordinates.

Each detected hand contains:

```text
21 landmarks
×
3 coordinates (x, y, z)
=
63 features
```

The processed data therefore has the following general structure:

```text
label | x0 | y0 | z0 | x1 | y1 | z1 | ... | x20 | y20 | z20
```

The classes used in this project are:

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

### Dataset source

If you redistribute the original image dataset, check the original dataset's license and usage conditions.

If the original dataset cannot legally be redistributed with this repository, upload only the processed data that you are permitted to distribute and provide the original dataset source/instructions instead.

**Dataset source:** `ADD_THE_ORIGINAL_DATASET_URL_HERE`

---

# 4. Important Dataset Consideration

The dataset contains samples for both static and potentially motion-dependent signs, but an image-based landmark classifier only sees one frame at a time.

This means that:

- Static gestures can be classified directly.
- Dynamic gestures cannot be fully represented by a single frame.
- Gestures such as `J` and `Z` may involve movement depending on the dataset/signing convention.
- A temporal model such as LSTM/GRU/Transformer is more appropriate when motion is essential.

This repository therefore treats the current SVM pipeline as a **static fingerspelling recognizer**.

---

# 5. Repository Structure

Recommended repository structure:

```text
ASL-Fingerspelling-Recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── <original dataset>
│   │
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── models/
│   ├── linear_svm_raw.pkl
│   ├── raw_scaler.pkl
│   ├── rbf_svc.pkl
│   └── raw_linear_svm.pkl
│
├── src/
│   ├── config.py
│   ├── extract_landmarks.ipynb
│   ├── Preprocess.ipynb
│   ├── train_model.ipynb
│   ├── evaluate_model.ipynb
│   ├── realtime_prediction_test.ipynb
│   └── fingerspelling_translator.ipynb
│
└── outputs/
    ├── confusion_matrices/
    └── figures/
```

You may change the folder names, but the paths in `config.py` must match the actual repository structure.

---

# 6. Environment

The project was developed using:

```text
Python 3.11.x
```

A virtual environment is strongly recommended.

## Windows

Create the environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 7. Install Dependencies

Install the required packages:

```bash
pip install numpy
pip install pandas
pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install joblib
pip install jupyter
```

Or install everything using:

```bash
pip install -r requirements.txt
```

A compatible NumPy version may be necessary depending on the installed MediaPipe version. If MediaPipe produces compatibility errors with NumPy 2.x, use:

```bash
pip install "numpy<2"
```

The project was developed and tested around Python 3.11.

---

# 8. Verify the Environment

Run:

```python
import numpy
import pandas
import cv2
import mediapipe
import sklearn
import matplotlib
import seaborn
import joblib

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mediapipe.__version__)
print("Scikit-learn:", sklearn.__version__)

print("Environment is ready.")
```

If all imports work without errors, continue with the pipeline.

---

# 9. Step 1 — Configure Paths

Open:

```text
src/config.py
```

Configure the dataset and output paths.

The important paths are:

```text
Raw dataset
Processed training data
Processed validation data
Processed test data
Models
```

Keep paths relative to the project root where possible so another user can clone the repository and reproduce the project without changing hard-coded personal paths.

Example:

```python
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
```

---

# 10. Step 2 — Extract Hand Landmarks

Open:

```text
src/extract_landmarks.ipynb
```

This notebook processes the original images using MediaPipe Hands.

For every image:

```text
Image
  ↓
MediaPipe Hands
  ↓
Hand detected
  ↓
21 landmarks
  ↓
x, y, z coordinates
  ↓
63 features
```

The resulting CSV contains:

```text
label + 63 landmark features
```

Run all cells in order.

If an image does not contain a detectable hand, it may be skipped.

After processing, verify the number of successful and skipped samples.

---

# 11. Step 3 — Preprocess the Landmark Data

Open:

```text
src/Preprocess.ipynb
```

This notebook prepares the landmark data for machine learning.

The labels are separated from the features:

```python
X = df.drop(columns=["label"])
y = df["label"]
```

The feature matrix has:

```text
63 features
```

---

## 11.1 Raw Landmarks

The raw landmark representation keeps the MediaPipe coordinates directly.

Pipeline:

```text
Raw landmarks
      ↓
StandardScaler
      ↓
Linear SVM
```

---

## 11.2 Wrist-Centered Landmarks

For wrist-centered normalization, landmark 0 (the wrist) is used as the origin.

For each sample:

```text
Every landmark
      -
Wrist coordinate
```

This makes the representation less dependent on where the hand appears in the image.

The transformation is:

```python
def wrist_center(X):

    X = X.reshape(-1, 21, 3)

    wrist = X[:, 0, :]

    X = X - wrist[:, np.newaxis, :]

    return X.reshape(-1, 63)
```

The resulting shape remains:

```text
(number_of_samples, 63)
```

---

# 12. Step 4 — Train the Models

Open:

```text
src/train_model.ipynb
```

The project currently contains three important SVM approaches.

## Model 1 — Raw Linear SVM

```text
Raw landmarks
     ↓
StandardScaler
     ↓
Linear SVM
```

## Model 2 — Wrist-Centered Linear SVM

```text
Raw landmarks
     ↓
Wrist-centered normalization
     ↓
StandardScaler
     ↓
Linear SVM
```

## Model 3 — RBF SVM

```text
Wrist-centered landmarks
     ↓
StandardScaler
     ↓
RBF SVM
```

Example:

```python
from sklearn.svm import SVC

linear_svm = SVC(
    kernel="linear",
    C=1,
    probability=True
)
```

`probability=True` is enabled because the real-time application displays an estimated prediction confidence using `predict_proba()`.

---

# 13. SVM Parameters

### `kernel`

For the linear model:

```python
kernel="linear"
```

This creates a linear decision boundary.

For the RBF model:

```python
kernel="rbf"
```

This allows nonlinear decision boundaries.

### `C`

```python
C=1
```

Controls the trade-off between:

- allowing classification errors
- keeping the decision boundary regularized

### `probability`

```python
probability=True
```

Enables probability estimates through:

```python
model.predict_proba(X)
```

This is useful for displaying confidence in the real-time application.

---

# 14. Step 5 — Save the Models

After training, save the trained models and scalers:

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

The scaler is part of the trained pipeline and must be saved.

You cannot train using one scaler and then use a different scaler during real-time inference.

---

# 15. Step 6 — Evaluate the Models

Open:

```text
src/evaluate_model.ipynb
```

The validation dataset is used to compare models during development.

For example:

```python
y_val_pred = model.predict(X_val_scaled)
```

Then calculate accuracy:

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

The evaluation notebook should also generate:

- Classification reports
- Precision
- Recall
- F1-score
- Confusion matrices

---

# 16. Why Accuracy Is Not Enough

Accuracy tells you the overall percentage of correctly classified samples.

For example:

```text
98.48%
```

means approximately 98.48 out of every 100 validation samples were classified correctly.

However, for an ASL classifier, per-class performance is also important.

A model can have high overall accuracy while struggling with particular letters.

Therefore, inspect:

```text
Precision
Recall
F1-score
Confusion Matrix
```

especially for confusing classes such as:

```text
N
S
T
X
Z
```

---

# 17. Current Validation Results

The current experiments produced approximately:

```text
Wrist-Centered Linear SVM
Validation Accuracy: 98.48%

RBF SVM
Validation Accuracy: 98.45%

Raw Linear SVM
Validation Accuracy: 72.29%
```

The wrist-centered representation therefore performed substantially better than the raw landmark representation.

The wrist-centered Linear SVM was particularly strong for the current static gesture task.

The RBF model achieved a very similar overall accuracy but showed different class-level behavior.

These numbers are **validation results**, not a guarantee of real-world webcam performance.

---

# 18. Step 7 — Test the Models in Real Time

Open:

```text
src/realtime_prediction_test.ipynb
```

This notebook loads the webcam and performs:

```text
Webcam
   ↓
MediaPipe
   ↓
21 landmarks
   ↓
63 features
   ↓
Scaler
   ↓
SVM
   ↓
Prediction
```

The frame used for ML processing should remain unflipped.

The frame can then be horizontally flipped only for visualization:

```python
display_frame = cv2.flip(frame, 1)
```

This gives the user a natural mirror-like webcam view without changing the landmark data used by the classifier.

---

# 19. Switching Models in Real Time

The real-time notebook supports switching between models.

```text
1 → Wrist-Centered Linear SVM
2 → RBF SVM
3 → Raw Linear SVM
Q → Quit
```

This makes it easy to compare the models using the same webcam input.

The prediction history is cleared when switching models to prevent predictions from the previous model affecting the new model.

---

# 20. Step 8 — Run the Fingerspelling Translator

Open:

```text
src/fingerspelling_translator.ipynb
```

This builds on the real-time recognizer.

Instead of only displaying:

```text
Prediction: A
```

the system builds:

```text
Text: ABC
```

---

## Gesture-to-letter interaction

The current interaction design is intentionally simple:

```text
Show gesture
      ↓
Hold gesture
      ↓
Prediction stabilizes
      ↓
Remove hand
      ↓
Letter is added
```

For example:

```text
Show H
Remove hand
→ H

Show E
Remove hand
→ HE

Show L
Remove hand
→ HEL

Show L
Remove hand
→ HELL

Show O
Remove hand
→ HELLO
```

This also allows repeated letters:

```text
L → release → L → release
```

to produce:

```text
LL
```

---

# 21. Prediction Stabilization

Individual webcam frames can occasionally be misclassified.

To reduce this effect, predictions are stored in a short history:

```python
prediction_history = deque(maxlen=10)
```

A majority vote is then used:

```python
stable_label = Counter(
    prediction_history
).most_common(1)[0][0]
```

Instead of trusting a single frame:

```text
Frame 1 → A
Frame 2 → A
Frame 3 → B
Frame 4 → A
Frame 5 → A
```

the system can determine:

```text
Stable prediction → A
```

This is useful for real-time inference.

---

# 22. Special Classes

The classifier contains two non-letter classes:

```text
space
del
```

### `space`

When the user performs the `space` gesture and releases the hand:

```text
HELLO
+
space
```

becomes:

```text
HELLO 
```

### `del`

When the user performs `del` and releases the hand:

```text
HELLO
+
del
```

becomes:

```text
HELL
```

---

# 23. Recommended Execution Order

For a completely fresh setup, run the notebooks in this order:

```text
1. config.py
       ↓
2. extract_landmarks.ipynb
       ↓
3. Preprocess.ipynb
       ↓
4. train_model.ipynb
       ↓
5. evaluate_model.ipynb
       ↓
6. realtime_prediction_test.ipynb
       ↓
7. fingerspelling_translator.ipynb
```

Do not skip directly to the real-time notebook unless the required processed data, models, and scalers already exist.

---

# 24. Reproducing the Project

A new user should be able to reproduce the project using:

```bash
git clone <YOUR_REPOSITORY_URL>
cd ASL-Fingerspelling-Recognition
```

Create an environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then:

```text
1. Download/prepare the dataset
2. Configure paths
3. Run extract_landmarks.ipynb
4. Run Preprocess.ipynb
5. Run train_model.ipynb
6. Run evaluate_model.ipynb
7. Run realtime_prediction_test.ipynb
8. Run fingerspelling_translator.ipynb
```

---

# 25. Webcam Requirements

The real-time notebooks require:

- A working webcam
- OpenCV camera access
- MediaPipe Hands
- A trained model
- The corresponding scaler
- The correct class mapping

If OpenCV cannot access the camera, check:

```python
cap = cv2.VideoCapture(0)
```

If necessary, try:

```python
cap = cv2.VideoCapture(1)
```

depending on the number assigned to the camera by the operating system.

---

# 26. Common Problems

## MediaPipe import problems

If MediaPipe fails because of NumPy compatibility, try:

```bash
pip install "numpy<2"
```

Then restart the Jupyter kernel.

---

## OpenCV camera error

If:

```python
success, frame = cap.read()
```

returns `False`, check camera permissions and whether another application is already using the webcam.

---

## `cvtColor` error

An error such as:

```text
(-215:Assertion failed) !_src.empty()
```

usually means OpenCV received an empty frame.

Check:

```python
if not success:
    break
```

before calling:

```python
cv2.cvtColor(...)
```

---

## Wrong predictions after changing preprocessing

The scaler used during inference must correspond to the preprocessing used during training.

For example:

```text
Wrist-centered training
        ↓
wrist-centered inference
        ↓
same scaler
        ↓
same model
```

Do not mix:

```text
Raw data + wrist scaler
```

or:

```text
Wrist-centered data + raw scaler
```

---

# 27. Important Model Rule

A scaler is **not a model**.

For example:

```python
raw_scaler
```

is a preprocessing object.

While:

```python
linear_svm_model
```

is the classifier.

The real-time pipeline is:

```text
Landmarks
   ↓
raw_scaler
   ↓
linear_svm_model
   ↓
Prediction
```

For a different trained pipeline:

```text
Wrist-centered landmarks
   ↓
wrist_scaler
   ↓
wrist-centered SVM
   ↓
Prediction
```

The preprocessing must match the model that was trained with it.

---

# 28. Limitations

The current system has several limitations.

### Static gesture assumption

The SVM receives one frame at a time.

It therefore does not explicitly model motion.

### Real-world lighting

Performance may change with:

- lighting
- camera quality
- background
- hand orientation
- distance from camera
- partial occlusion

### Dataset bias

High validation accuracy on a dataset does not necessarily mean equivalent performance with a webcam.

### Dynamic signs

A temporal model is more suitable for gestures whose meaning depends on movement.

Potential future approaches include:

```text
Landmark sequences
      ↓
LSTM / GRU
      ↓
Dynamic gesture classification
```

or:

```text
Landmark sequences
      ↓
Transformer
      ↓
Temporal gesture recognition
```

---

# 29. Future Improvements

Possible future development:

- Better temporal gesture handling
- LSTM/GRU sequence model
- Transformer-based temporal model
- Two-hand ASL recognition
- Improved gesture segmentation
- Automatic letter acceptance
- Gesture cooldown/debounce
- Word-level recognition
- Language-model-based sentence correction
- Text-to-speech
- Mobile deployment
- Edge deployment
- Larger and more diverse datasets
- Better handling of `J` and `Z`
- Real-world evaluation with different users

---

# 30. Project Results

The current SVM experiments demonstrate that landmark-based classification can achieve strong validation performance for static ASL fingerspelling.

The most successful representation in the current experiments was:

```text
MediaPipe landmarks
        ↓
Wrist-centered normalization
        ↓
StandardScaler
        ↓
Linear SVM
```

with approximately:

```text
98.48% validation accuracy
```

The RBF SVM produced a very similar result:

```text
98.45% validation accuracy
```

The raw landmark Linear SVM performed considerably worse:

```text
72.29% validation accuracy
```

This highlights the importance of choosing an appropriate landmark representation before increasing model complexity.

---

# 31. License

Add the license appropriate for your project here.

For example:

```text
MIT License
```

However, make sure the license of the repository is compatible with the license of any dataset, pretrained model, or third-party component you redistribute.

---

# 32. Acknowledgements

This project uses:

- MediaPipe Hands for hand landmark detection
- OpenCV for computer vision and webcam processing
- Scikit-learn for machine-learning models
- NumPy for numerical processing
- Pandas for data processing
- Matplotlib / Seaborn for evaluation visualization
- Joblib for model serialization

---

# 33. Author

**Shehab Abdo**

Mechatronics & Robotics Engineer

GitHub: `ADD_YOUR_GITHUB_PROFILE`

Portfolio: `ADD_YOUR_PORTFOLIO_URL`

---

# 34. Quick Start

For experienced users:

```bash
git clone <YOUR_REPOSITORY_URL>
cd ASL-Fingerspelling-Recognition

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Then open Jupyter:

```bash
jupyter notebook
```

Run:

```text
src/extract_landmarks.ipynb
        ↓
src/Preprocess.ipynb
        ↓
src/train_model.ipynb
        ↓
src/evaluate_model.ipynb
        ↓
src/realtime_prediction_test.ipynb
        ↓
src/fingerspelling_translator.ipynb
```

Enjoy building with ASL landmark-based recognition!
