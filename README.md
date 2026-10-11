# Brain Tumor Detection Using Computer Vision

A deep learning project that classifies brain MRI images into four categories using **VGG16 transfer learning, TensorFlow, and Flask**. The project provides a web interface where users can upload an MRI image and view the predicted class and confidence score.

## Project Overview

Brain tumor classification from MRI images is an important computer vision application. This project explores how a pretrained convolutional neural network (CNN) can be adapted to classify brain MRI images and how the trained model can be integrated into a web application.

The project combines a Jupyter Notebook for model training and evaluation with a Flask application for image upload and prediction.

## Objectives

- Classify brain MRI images into four categories.
- Apply transfer learning using the pretrained VGG16 architecture.
- Preprocess and normalize MRI images before inference.
- Evaluate the model using classification metrics and visualizations.
- Integrate the trained model into a Flask web application.
- Display the predicted class and model confidence score through a simple user interface.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development and inference |
| VGG16 | Pretrained CNN for transfer learning |
| NumPy | Numerical operations and image-array processing |
| Pillow (PIL) | Image loading and preprocessing |
| Scikit-learn | Classification report, confusion matrix, and ROC analysis |
| Matplotlib / Seaborn | Data visualization and evaluation plots |
| Flask | Web application and prediction interface |
| HTML | Frontend user interface |
| Git / Git LFS | Version control and large model-file storage |

## Dataset and Classification Categories

The dataset is organized into separate training and testing directories, with images grouped by class.

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The dataset is excluded from this GitHub repository to keep the repository size manageable.

### The four classes

| Class | Description |
|---|---|
| Glioma | MRI images labelled as glioma |
| Meningioma | MRI images labelled as meningioma |
| Pituitary | MRI images labelled as pituitary tumor |
| No Tumor | MRI images labelled as having no tumor |

The model predicts one of these four classes based on the input image.

## Model Architecture

The project uses **VGG16 with transfer learning**. VGG16 is a CNN pretrained on ImageNet, and its convolutional feature extractor is reused for MRI image classification.

The notebook configures the model as follows:

1. **Input:** MRI images resized to 128 × 128 pixels with three color channels.
2. **Pretrained base:** VGG16 with ImageNet weights and its original classification head removed.
3. **Fine-tuning:** Most VGG16 layers are frozen, while the last few layers are made trainable.
4. **Flatten:** Converts the extracted feature maps into a one-dimensional feature vector.
5. **Dropout:** Dropout layers with rates of 0.3 and 0.2 help reduce overfitting.
6. **Dense layer:** A fully connected layer with 128 neurons and ReLU activation.
7. **Output layer:** Four neurons with softmax activation for four-class classification.

### Training configuration

-Dataset split	Number of images
-Training	5,712
-Testing	1,311
-Total	7,023
- Image size: 128 × 128
- Batch size: 20
- Number of epochs: 5
- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: Sparse categorical cross-entropy
- Training metric: Sparse categorical accuracy

These are the configuration values specified in the notebook; they are not a claim of optimal performance.

## Image Preprocessing

The training notebook and Flask application perform image preprocessing before passing images to the model.

- Resize the input image to 128 × 128 pixels.
- Convert the image into a numerical array.
- Normalize pixel values to the range [0, 1].
- Add a batch dimension before inference.

The notebook also applies random brightness and contrast adjustments as image augmentation during data preparation.

## Model Evaluation

The notebook includes several methods for evaluating classification performance:

- **Classification report:** Precision, recall, and F1-score for each class.
- **Confusion matrix:** Compares actual labels with predicted labels.
- **ROC curve:** Visualizes the true-positive rate against the false-positive rate.
- **AUC:** Summarizes ROC performance for each class.
- **Training history:** Visualizes training accuracy and loss.

Actual accuracy and other performance values should be reported only after running the evaluation and verifying the results.

## Web Application Features

The Flask application provides a simple interface for making predictions on uploaded MRI images.

- Upload an MRI image through the web interface.
- Preprocess the uploaded image automatically.
- Load the saved Keras model and perform inference.
- Display the predicted category.
- Display the model's confidence score.
- Show the uploaded image alongside the result.

For a prediction labelled `notumor`, the application displays **No Tumor**. For the other classes, it displays the corresponding predicted tumor category.

## Project Structure

```text
Brain-Tumor-Detection-Using-Computer-Vision/
│
├── main.py
├── model.h5
├── brain_tumour_detection_using_deep_learning.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── templates/
│   └── index.html
│
├── Te-gl_0015.jpg
├── Te-meTr_0001.jpg
├── Te-noTr_0004.jpg
└── Te-piTr_0003.jpg
```

**Key files**

- `main.py` — Flask application and prediction logic.
- `model.h5` — Saved trained classification model, stored using Git LFS.
- `brain_tumour_detection_using_deep_learning.ipynb` — Notebook containing dataset preparation, model training, evaluation, and example predictions.
- `templates/index.html` — HTML frontend.
- `requirements.txt` — Python dependency list.
- `Te-gl_0015.jpg`, `Te-meTr_0001.jpg`, `Te-noTr_0004.jpg`, `Te-piTr_0003.jpg` — Example MRI images.

The complete dataset and temporary uploaded images are excluded from the repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vijay7300/Brain-Tumor-Detection-Using-Computer-Vision.git
cd Brain-Tumor-Detection-Using-Computer-Vision
```

### 2. Set up Git LFS

The trained model is stored using Git Large File Storage (LFS).

Install Git LFS from https://git-lfs.com/ if it is not already installed, then run:

```bash
git lfs install
git lfs pull
```

Verify that `model.h5` is the actual model file and not just a small LFS pointer file.

### 3. Create a virtual environment

Using Python on Windows:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can invoke the environment's Python directly using `.\venv\Scripts\python.exe`.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python and TensorFlow combination compatible with the versions in `requirements.txt`.

### 5. Run the application

```bash
python main.py
```

Open the following address in your browser:

http://127.0.0.1:5000

Upload an MRI image to view the model's prediction.

## How the Application Works

```text
MRI Image Upload
       |
       v
Image Preprocessing
(128 × 128, Normalization)
       |
       v
Trained VGG16-Based Model
       |
       v
Four-Class Prediction
       |
       v
Predicted Category
+ Confidence Score
       |
       v
Result Displayed in Flask UI
```

## Future Improvements

- Validate the class-index mapping between training and inference.
- Evaluate the model on an independent test set and report verified metrics.
- Improve generalization using suitable augmentation and regularization.
- Add input validation and secure file-upload handling.
- Improve the frontend and display clearer prediction results.
- Deploy the application to a suitable hosting platform.
- Investigate model explainability methods, such as Grad-CAM, to visualize image regions influencing predictions.

## Limitations and Medical Disclaimer

This project is intended for educational and research purposes only. It has not been established as a clinically validated medical diagnostic tool.

Predictions depend on the training data, image quality, model limitations, and preprocessing. A confidence score is not necessarily a calibrated probability of a correct diagnosis. The application must not be used to diagnose, exclude, or guide treatment for a brain tumor. Medical concerns should be assessed by qualified healthcare professionals.

## Author

**Vijay Prajapati**

M.Sc-M.Tech — Data & Computational Sciences, IIT Jodhpur
Date:- 11-10-2026
GitHub: [Vijay7300](https://github.com/Vijay7300)

---

If you find this project useful, feel free to explore the code, learn from the implementation, and suggest improvements.
