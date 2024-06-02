# Heart Disease Prediction

## Overview

This project aims to predict heart disease using the UCI Heart Disease dataset. The dataset contains various medical features that help in predicting whether a person has heart disease or not. The project involves data preprocessing, visualization, and the application of multiple machine learning models to achieve the highest prediction accuracy.

## Directory Structure

```
.
├── heart.csv           # Dataset file
├── main.py             # Main script file
└── README.md           # Readme file (this file)
```

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using the following command:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Data Description

The dataset consists of 303 rows and 14 columns. Each row represents a patient, and each column represents a specific attribute:

- `age`: Age of the patient
- `sex`: Sex of the patient (1 = male; 0 = female)
- `cp`: Chest pain type (0, 1, 2, 3, 4)
- `trtbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- `restecg`: Resting electrocardiographic results (values 0, 1, 2)
- `thalachh`: Maximum heart rate achieved
- `exng`: Exercise-induced angina (1 = yes; 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slp`: Slope of the peak exercise ST segment
- `caa`: Number of major vessels (0-3) colored by fluoroscopy
- `thall`: Thalassemia (0 = normal; 1 = fixed defect; 2 = reversable defect)
- `output`: Diagnosis of heart disease (1 = presence; 0 = absence)

## Data Preprocessing

1. **Reading Data**: The dataset is read into a pandas DataFrame.
2. **Descriptive Statistics**: Basic statistics of the dataset are displayed.
3. **Data Information**: Data types and non-null counts are displayed.
4. **Missing Values**: The dataset is checked for missing values.
5. **Categorical and Numerical Features**: The dataset is divided into categorical and numerical features for separate processing.

## Data Visualization

Various visualizations are created to understand the data better:
- Count plots for categorical features with respect to the target variable (`output`).
- Pair plots for numerical features colored by the target variable.
- Box plots and swarm plots for numerical features.
- Correlation heatmap for numerical features.

## Outlier Detection and Removal

Outliers in numerical features are detected using the Interquartile Range (IQR) method and removed to improve model performance.

## Feature Scaling and Encoding

- Numerical features are scaled using `StandardScaler`.
- Categorical features are one-hot encoded.

## Model Training and Evaluation

Three models are trained and evaluated on the dataset:

1. **Logistic Regression**
   - Standard logistic regression.
   - Hyperparameter tuning using `GridSearchCV` for the penalty parameter.

2. **K-Nearest Neighbors (KNN)**
   - KNN with K=3.

3. **Support Vector Classifier (SVC)**
   - Standard SVC.

Each model is evaluated using accuracy, confusion matrix, and classification report.

## How to Run

1. Ensure all required packages are installed.
2. Place the dataset (`heart.csv`) in the same directory as the script.
3. Run the main script:

```sh
python main.py
```

The script will output the following:
- Accuracy scores for each model.
- ROC curve for the logistic regression model.
- Confusion matrices and classification reports for all models.

## Conclusion

The project demonstrates the application of various machine learning techniques to predict heart disease. The logistic regression model with hyperparameter tuning achieved the highest accuracy.

## Future Work

- Explore more advanced models like Random Forest, Gradient Boosting, etc.
- Perform cross-validation for more robust model evaluation.
- Experiment with different feature engineering techniques.

---

For any questions or suggestions, feel free to reach out.

