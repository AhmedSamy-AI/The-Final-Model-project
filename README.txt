# Brain Tumor Detection using MLP

1. Project Overview

---

This project is an educational brain MRI image classification project using a Multi-Layer Perceptron (MLP) neural network.

The model classifies brain MRI images into two classes:

0 -> No Tumor / Healthy
1 -> Tumor

The project includes a Jupyter Notebook for data preprocessing, model training, evaluation, and analysis, as well as a Streamlit web application for making predictions on new MRI images.

Important:
This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and must not be used for real clinical decisions.

2. Project Structure

---

MODEL/
|
|-- app.py
|-- README.txt
|-- Requirements.txt
|
`-- src/     |-- notebook.ipynb
    `-- mlp_model.keras

3. Dataset

---

The dataset contains two classes:

yes -> Tumor MRI images
no  -> Healthy MRI images

The images are assigned the following labels:

Tumor   = 1
Healthy = 0

Exact duplicate images are detected and removed before training.

4. Image Preprocessing

---

Each image is:

* Converted to grayscale.
* Resized to 64 x 64 pixels.
* Converted to float32.
* Normalized by dividing pixel values by 255.
* Given a channel dimension before being passed to the model.

The dataset is split into:

80% Training
20% Validation

The split uses:

test_size = 0.2
random_state = 42
shuffle = True
stratify = y_all

Class weights are also used during training to help handle class imbalance.

5. Data Augmentation

---

Training data uses data augmentation techniques including:

* Random horizontal flipping.
* Random rotation of approximately +/- 10 degrees.

These techniques help improve the model's generalization.

6. MLP Model Architecture

---

Input:
64 x 64 grayscale image

Architecture:

Input (64 x 64 x 1)
|
Flatten
|
Dense (256, ReLU)
|
Batch Normalization
|
Dropout (0.3)
|
Dense (128, ReLU)
|
Batch Normalization
|
Dropout (0.3)
|
Dense (1, Sigmoid)

L2 regularization is applied to the Dense layers to help reduce overfitting.

7. Model Training

---

Optimizer:
Adam

Loss:
Binary Crossentropy

Metrics:
Accuracy
AUC
Precision
Recall

Batch Size:
32

Maximum Epochs:
60

Class Weights:
Used during training to handle class imbalance.

Callbacks:

* EarlyStopping is used to stop training when validation performance stops improving and restore the best model weights.
* ReduceLROnPlateau is used to reduce the learning rate when validation loss stops improving.

8. Model Evaluation

---

The model is evaluated using:

* Validation Loss
* Accuracy
* AUC
* Precision
* Recall

The notebook also generates:

* Training and Validation Accuracy plots.
* Training and Validation Loss plots.
* Confusion Matrix.
* Classification Report.

9. Threshold Analysis

---

The notebook evaluates different classification thresholds between 0.10 and 0.90.

For each threshold, the following metrics are calculated:

* Precision
* Recall
* F1 Score

The threshold with the highest F1 Score is identified as the Best F1 Threshold.

This threshold can be used in the Streamlit application instead of the default threshold of 0.5 when desired.

10. Model Saving

---

After training, the trained model is saved as:

src/mlp_model.keras

The Streamlit application loads this saved model for prediction.

11. Streamlit Application

---

The Streamlit application is located in:

app.py

The application allows the user to:

1. Upload a brain MRI image.
2. Preview the uploaded image.
3. Convert the image to grayscale.
4. Resize the image to 64 x 64 pixels.
5. Normalize the pixel values.
6. Pass the image to the trained MLP model.
7. Display the prediction score.
8. Display the predicted result.

The application accepts:

.jpg
.jpeg
.png

Possible results:

Tumor Detected
No Tumor Detected

12. Requirements

---

Python Version:
Python 3.13

The required Python packages are listed in:

Requirements.txt

Main dependencies include:

TensorFlow
Streamlit
OpenCV
NumPy
Pillow

13. How to Run the Application

---

Step 1:
Open Command Prompt and navigate to the project folder.

Example:

cd C:\Users\Ahmed\Desktop\MODEL

Step 2:
Install the required packages:

py -3.13 -m pip install -r Requirements.txt

Step 3:
Run the Streamlit application:

py -3.13 -m streamlit run app.py

The application will open in the browser.

If it does not open automatically, use:

http://localhost:8501

14. How to Stop the Application

---

Return to the Command Prompt and press:

Ctrl + C

15. Notebook Workflow

---

The Jupyter Notebook contains the complete machine learning workflow:

1. Dataset loading.

2. Image preprocessing.

3. Duplicate detection and removal.

4. Train/validation splitting.

5. Class weight calculation.

6. Data augmentation.

7. MLP model construction.

8. Model training.

9. Model saving.

10. Model evaluation.

11. Accuracy and loss visualization.

12. Confusion Matrix.

13. Classification Report.

14. Threshold analysis.

15. Important Note

---

This project is intended for educational and demonstration purposes only.

The predictions produced by the model are not medical diagnoses and should not be used for real clinical decisions.
