# 🌍 Seismic Activity Classification & Analysis using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A comprehensive Deep Learning solution designed to analyze and classify seismic activity data. This project leverages Artificial Neural Networks (ANN) and modern Data Science workflows to predict seismic events and pattern anomalies.

---

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Dataset Overview](#dataset-overview)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Tech Stack & Libraries](#tech-stack--libraries)
- [Installation & Local Setup](#installation--local-setup)
- [Results & Evaluation](#results--evaluation)
- [License](#license)

---

## 🎯 About the Project

Seismic activity monitoring is crucial for early hazard detection and geological assessment. Traditional Machine Learning models (such as Support Vector Machines) often require heavy feature engineering to capture non-linear geological relationships. 

This repository implements a **Deep Learning approach** to process seismic feature measurements, perform Exploratory Data Analysis (EDA), handle class imbalance/scaling, and classify seismic activity patterns using Deep Neural Networks.

---

## 📊 Dataset Overview

The dataset used in this project is stored in `08-seismic_activity_svm`. It contains geophysical attributes recorded during seismic monitoring sessions:

- **Features:** Geophysical attributes (energy readings, pulse frequencies, shift factors, structural variables).
- **Target Variable:** Binary/Categorical classification indicating seismic hazard or activity state.
- **Preprocessing:** 
  - Missing value checks and imputation.
  - Feature normalization & standardization via `StandardScaler`.
  - Train/Test split for robust validation.

---

## 🛠️ Project Workflow

1. **Exploratory Data Analysis (EDA):**
   - Summary statistics (`describe().T`), missing data detection (`isnull().sum()`), and feature distribution analysis.
   - Outlier detection and correlation matrix evaluation.

2. **Data Preprocessing & Feature Engineering:**
   - Feature scaling to optimize gradient descent convergence in deep neural layers.
   - Categorical encoding (if applicable) and target vector formatting.

3. **Deep Learning Model Building:**
   - Multi-Layer Perceptron (MLP) / Feedforward Neural Network architecture.
   - Integration of `Dropout` layers to prevent overfitting.
   - Optimization using modern optimizers (`Adam`) and dynamic loss tracking.

4. **Model Evaluation:**
   - Performance evaluation using Loss vs. Epochs curves, Accuracy metrics, Confusion Matrix, and Classification Report.

---

## 🧠 Model Architecture

The core deep learning model implemented in `Seismic_DL_Model.ipynb` follows a sequential neural network pipeline:

[ Input Layer ]  ---> Dense Layer (ReLU + BatchNorm)
---> Dropout Layer (Regularization)
---> Dense Layer (ReLU)
---> Output Layer (Sigmoid / Softmax)


- **Loss Function:** Binary Cross-Entropy / Categorical Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy, Precision, Recall, F1-Score

---

## 💻 Tech Stack & Libraries

- **Language:** Python 3.x
- **Deep Learning Framework:** TensorFlow / Keras (or PyTorch)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

---

## 🚀 Installation & Local Setup

To run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/emrezorlu1239/seismic-activity-deep-learning.git](https://github.com/emrezorlu1239/seismic-activity-deep-learning.git)
   cd seismic-activity-deep-learning
Create a Virtual Environment (Optional but Recommended):

Bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install Dependencies:

Bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
Launch Jupyter Notebook:

Bash
jupyter notebook Seismic_DL_Model.ipynb
## 📈 Results & Evaluation
High accuracy and generalization on test seismic sequences.

Loss curves demonstrate stable learning rates without severe overfitting.

Detailed metrics can be inspected directly inside the Seismic_DL_Model.ipynb notebook.

## 📜 License
Distributed under the MIT License. See LICENSE for more information.
