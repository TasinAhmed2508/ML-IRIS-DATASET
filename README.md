# 🌸 Iris Dataset Classification - ML Algorithms Comparison





















SOFTWARE.OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THELIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHERFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THEIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS ORcopies or substantial portions of the Software.The above copyright notice and this permission notice shall be included in allfurnished to do so, subject to the following conditions:copies of the Software, and to permit persons to whom the Software isto use, copy, modify, merge, publish, distribute, sublicense, and/or sellin the Software without restriction, including without limitation the rightsof this software and associated documentation files (the "Software"), to dealPermission is hereby granted, free of charge, to any person obtaining a copyCopyright (c) 2026 Tasin Ahmed![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
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

## 📈 Performance Metrics

For detailed performance analysis and comparison of all algorithms, see [ANALYSIS.md](ANALYSIS.md).

## 👨‍💻 Author

**Tasin Ahmed**

Connect with me:
- GitHub: [@TasinAhmed2508](https://github.com/TasinAhmed2508)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2026 Tasin Ahmed

---

*Built with ❤️ for the data science community*

This project is open source and available under the MIT License.

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Iris Dataset on UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

**Last Updated:** January 2026

