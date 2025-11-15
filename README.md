# Machine Learning Classification & Model Evaluation

This project implements a comprehensive end‑to‑end machine learning workflow spanning data preprocessing, model training, cross‑validation, algorithm benchmarking, feature selection, cost‑sensitive learning, and large‑scale predictive modeling. Multiple datasets from medical, credit approval, radar ionosphere, and UCI repositories are used to evaluate a variety of supervised learning algorithms under different experimental conditions.

The project demonstrates practical competency in modern machine learning using **Scikit‑learn**, with a strong focus on reproducible experimentation, data quality management, and systematic model comparison.

## Overview of Skills & Techniques

This project highlights proficiency in the following core machine learning skills:

### **Data Cleaning & Preprocessing**

* Detection and handling of missing values
* Mean imputation for numerical features
* Mode imputation for categorical features
* Standardization/normalization using `StandardScaler`
* One‑hot encoding of categorical variables
* Training–test leakage prevention by computing statistics only on training data
* Dataset‑specific preprocessing (e.g., removing ID attributes, parsing raw `.data` files)

### **Feature Engineering & Selection**

* Mutual Information feature ranking (`mutual_info_classif`)
* Selection of top‑k informative attributes
* Impact analysis of feature reduction on classifier performance

### **Model Training & Evaluation**

* Training and tuning of multiple supervised learning algorithms
* Experiments with varying hyperparameters (e.g., decision tree splitting constraints)
* Comparison of model complexity vs. generalization
* Visualization of decision trees
* Performance measurement using accuracy and error rate
* Evaluation under multiple dataset distributions and sizes

### **Cross‑Validation & Statistical Evaluation**

* K‑fold cross‑validation across multiple datasets
* Ranking of algorithms by performance
* Interpretation of whether differences are statistically meaningful (paired t-test)

### **Handling Imbalanced Datasets**

* Cost‑sensitive learning using custom misclassification cost matrices
* Use of class weighting strategies
* Exploration of resampling methods (random over/under-sampling, SMOTE)

### **Large‑Scale Prediction Pipeline**

* Training models on 30,000+ samples
* Applying the final model to a 20,000‑record test set
* Generating class predictions for real‑world operational use
* Documenting the rationale for choices in preprocessing and modeling

## Classification Algorithms Used

The following supervised learning models were implemented, tuned, and evaluated throughout the project:

* **Decision Tree Classifier**
* **k‑Nearest Neighbors**
* **Gaussian Naive Bayes**
* **Logistic Regression**
* **MLP Neural Network**
* **Random Forest**

All stochastic models use `random_state` for reproducibility.
* Create a polished academic-style report based on this README
