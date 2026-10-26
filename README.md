# Brain-Tumor-MRI-Classification
Brain Tumor MRI Classification using a custom CNN and transfer learning with EfficientNetB0, ResNet50, and MobileNetV2, with model evaluation and performance comparison.   Copy description
# Brain Tumor MRI Classification Using Deep Learning

## Overview

This project explores the use of Deep Learning and Convolutional Neural Networks (CNNs) for classifying brain MRI images into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The project compares a custom CNN built from scratch with three pretrained architectures using transfer learning: **EfficientNetB0, ResNet50, and MobileNetV2**.

The goal is to experiment with different architectures, evaluate their performance on the same classification task, and understand how custom CNNs compare with pretrained models in medical image classification.

##  Project Objectives

- Build a custom CNN architecture from scratch.
- Apply transfer learning using pretrained CNN architectures.
- Experiment with fine-tuning pretrained models.
- Implement image preprocessing and data augmentation.
- Evaluate models using accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.
- Compare the performance of different architectures on a held-out test set.

##  Dataset

This project uses the **Brain Tumor MRI Dataset** by Masoud Nickparvar, available on Kaggle.

**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset contains four classes:

| Class | Description |
|---|---|
| Glioma | MRI images categorized as glioma |
| Meningioma | MRI images categorized as meningioma |
| No Tumor | MRI images categorized as having no tumor |
| Pituitary | MRI images categorized as pituitary tumor |

### Dataset Distribution

The original dataset contains 5,600 training images and 1,600 testing images.

| Split | Images |
|---|---:|
| Training subset | 4,760 |
| Validation subset | 840 |
| Testing set | 1,600 |
| **Total** | **7,200** |

The original training set was divided into training and validation subsets using an 85/15 stratified split, with a fixed random state of 42. The separate testing set was retained for final model evaluation.

Each class has 1,400 images in the original training set and 400 images in the testing set.

##  Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **OpenCV**
- **Pillow (PIL)**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

##  Methodology

### 1. Data Preprocessing

The dataset was prepared for model training using the following steps:

- Resizing images to **224 × 224** pixels.
- Rescaling pixel values to the range [0, 1].
- Loading images in batches of 32.
- Applying data augmentation to training images using random horizontal flips and random rotations.
- Using parallel data loading and prefetching to improve pipeline efficiency.

### 2. Custom CNN

A CNN architecture was developed from scratch to learn visual features directly from the MRI images.

The architecture includes:

- Convolutional layers with ReLU activation.
- Batch normalization.
- Max pooling.
- Dropout for regularization.
- Global average pooling.
- A dense layer with 256 units.
- A four-unit softmax output layer for classification.

### 3. Transfer Learning

Three pretrained CNN architectures were explored:

- **EfficientNetB0:** A computationally efficient CNN architecture.
- **ResNet50:** A deep residual network using skip connections.
- **MobileNetV2:** A lightweight architecture designed for computational efficiency.

Transfer learning was used to leverage pretrained feature representations, followed by additional training and fine-tuning experiments.

### 4. Training Strategy

The training process included:

- Adam optimizer.
- Categorical cross-entropy loss.
- Early stopping based on validation accuracy.
- Learning rate reduction when validation loss stopped improving.
- Model checkpointing to save the best-performing model during training.

### 5. Model Evaluation

All four models were evaluated on the same held-out test set of 1,600 images.

The evaluation included:

- Accuracy.
- Precision, recall, and F1-score.
- Confusion matrices.
- One-vs-rest ROC curves and AUC scores.
- A comparison of overall test accuracy.

##  Results

The following table summarizes the test accuracy obtained by each model in this experiment.

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 91.06% |
| EfficientNetB0 | 90.12% |
| ResNet50 | **94.31%** |
| MobileNetV2 | 81.69% |

### Key Findings

- **ResNet50 achieved the highest test accuracy of 94.31%** among the four evaluated models.
- The custom CNN achieved 91.06% accuracy, demonstrating that a model built from scratch can perform competitively on this dataset.
- EfficientNetB0 achieved 90.12% accuracy.
- MobileNetV2 achieved 81.69% accuracy in the evaluated configuration.
- The results highlight that model architecture and training configuration can influence classification performance.

These results represent performance on this specific dataset and experimental setup. They do not establish clinical reliability or generalization to other datasets or real-world medical settings.

##  Visualizations

The notebook includes visualizations to support model evaluation and comparison, including:

- Sample MRI images from the dataset.
- Confusion matrices for each model.
- ROC curves with class-specific AUC scores.
- A bar chart comparing test accuracy across the four architectures.

These visualizations help examine not only overall accuracy but also how each model performs across the four classes.

##  How to Run the Project

The project was developed in Kaggle Notebooks using TensorFlow/Keras.

### Option 1: Run on Kaggle

1. Open the [project notebook](https://www.kaggle.com/code/nourkhaledd/brain-tumor-mri).
2. Click **Copy & Edit** to create your own notebook.
3. Add the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) to your notebook.
4. Enable GPU acceleration in the notebook settings if available.
5. Run the notebook cells in order, from data preparation through model evaluation.

### Option 2: Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification
   ```

2. Install the required libraries:

   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
   ```

3. Download the dataset from Kaggle and extract it locally.

4. Update the dataset paths in the notebook to match your local directory structure.

5. Run the notebook using Jupyter Notebook or JupyterLab.

**Note:** The dataset path in the original notebook points to a Kaggle environment. It must be updated when running locally. The model weights are saved to the Kaggle working directory during execution and are not automatically included in the repository.


The notebook contains the complete workflow, including preprocessing, model building, training, evaluation, and comparison.

##  Future Improvements

Potential directions for further development include:

- Hyperparameter optimization for all architectures.
- More systematic comparison of transfer learning and fine-tuning strategies.
- External validation on an independent MRI dataset.
- Model interpretability using Grad-CAM or similar techniques.
- Testing robustness to variations in image quality and acquisition settings.
- Developing an inference interface for experimenting with individual MRI images.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is an experimental image classification project and **is not a medical diagnostic tool**. Its results should not be used to make clinical decisions or replace professional medical evaluation.

## 👩‍💻 Author

**Nour Khaled**

- Kaggle: [nourkhaledd](https://www.kaggle.com/nourkhaledd)
- Project Notebook: [Brain Tumor MRI Classification](https://www.kaggle.com/code/nourkhaledd/brain-tumor-mri)
