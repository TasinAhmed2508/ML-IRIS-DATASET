# 🌸 Iris Dataset Classification - ML Algorithms Comparison

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning project comparing 7 classification algorithms on the classic Iris dataset. This repository demonstrates best practices in data science, including exploratory data analysis, model training, evaluation, and performance comparison across multiple algorithms.

## 📊 Overview

This project addresses the **multi-class classification problem** of identifying iris flower species based on physical measurements. The solution implements and evaluates 7 different machine learning algorithms, providing insights into their strengths, weaknesses, and optimal use cases.

**Problem:** Classify iris flowers into three species (Setosa, Versicolor, Virginica) using sepal and petal measurements.

**Solution:** Systematic comparison of traditional ML algorithms with comprehensive evaluation metrics and visualizations.

## 🎯 Algorithms Implemented

1. **Decision Tree Classifier** - Interpretable tree-based model
2. **K-Nearest Neighbors (KNN)** - Distance-based classification
3. **Logistic Regression** - Linear probabilistic model
4. **Naive Bayes Classifier** - Probabilistic classification with independence assumption
5. **Random Forest Classifier** - Ensemble of decision trees
6. **Support Vector Machine (SVM)** - Kernel-based maximum margin classifier
7. **Gradient Boosting Classifier** - Sequential ensemble learning

## 📁 Folder Structure

```
Iris-Dataset/
│
├── notebooks/                           # Jupyter notebooks for analysis
│   ├── iris_decision_tree_classifier.ipynb
│   ├── iris_k_nearest_neighbors.ipynb
│   ├── iris_logistic_regression.ipynb
│   ├── iris_naive_bayes_classifier.ipynb
│   ├── iris_random_forest_classifier.ipynb
│   ├── iris_support_vector_machine.ipynb
│   └── iris_gradient_boosting.ipynb
│
├── Iris.csv                             # Dataset file
├── ANALYSIS.md                          # Detailed performance analysis
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
└── LICENSE                              # MIT License
```

## 📋 Dataset

**Name:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
**Source:** UCI Machine Learning Repository  
**Size:** 150 samples (50 per class)  
**Features:** 4 numerical attributes
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

**Target Variable:** Species (3 classes - Setosa, Versicolor, Virginica)

## 🏆 Key Results

- **Best Model:** Gradient Boosting Classifier
- **Accuracy:** ~99% on test set
- **Runner-ups:** Random Forest (98%), SVM (98%)
- **Fastest Model:** Logistic Regression (97% accuracy, lowest complexity)
- **Most Interpretable:** Decision Tree (97% accuracy, visual decision rules)

All models achieved >95% accuracy, demonstrating that the Iris dataset is well-suited for classification tasks.

## 🚀 Installation & Usage

### Clone Repository
```bash
git clone https://github.com/TasinAhmed2508/Iris-Dataset.git
cd Iris-Dataset
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Navigate to notebooks/ folder and open any .ipynb file
```

Each notebook contains:
- Data loading and exploration
- Model training and hyperparameter tuning
- Performance evaluation with metrics
- Visualization of results

## 🛠️ Technologies Used

- **Python 3.7+**
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **Jupyter** - Interactive notebooks

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iris-classification.git
cd iris-classification

# Install required packages
pip install -r requirements.txt
```

### Running Notebooks

Open any notebook in Jupyter:
```bash
jupyter notebook iris_decision_tree.ipynb
```

## 📈 Key Features

- ✅ Data loading and exploration
- ✅ Feature scaling and preprocessing
- ✅ Train-test split validation
- ✅ Model training and hyperparameter tuning
- ✅ Performance metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Confusion matrix visualization
- ✅ ROC curves and classification reports
- ✅ Feature importance analysis

## 💡 Learning Outcomes

This project demonstrates:
- How to implement various ML algorithms from scikit-learn
- Model evaluation and comparison techniques
- Data visualization best practices
- Hyperparameter tuning strategies
- Cross-validation methods

## 📊 Results

Each notebook includes detailed performance metrics and visualizations comparing:
- Classification accuracy
- Training time
- Model complexity
- Strengths and weaknesses of each approach

## 🤝 Contributing

Feel free to fork this repository and submit pull requests with improvements, additional algorithms, or enhanced visualizations.

## 📝 License

This project is open source and available under the MIT License.

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Iris Dataset on UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

**Last Updated:** January 2026

