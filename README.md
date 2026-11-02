# Naive-bays

## 📌 Project Overview
This project uses the **Gaussian Naive Bayes algorithm** to classify Iris flowers into three species: Setosa, Versicolor, and Virginica. The model is developed using Python and Scikit-learn and predicts flower species based on sepal and petal measurements.

## 🎯 Objectives
- Load and preprocess the Iris dataset.
- Convert flower species labels into numerical values.
- Split the dataset into training and testing sets.
- Train a Gaussian Naive Bayes classification model.
- Evaluate the model using performance metrics.
- Predict the species of a new flower.
- Save and load the trained model using Pickle.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pickle

## 📂 Dataset
The project uses the Iris dataset, which contains measurements of three flower species.

**Input Features:**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Target Classes:**
- Iris Setosa
- Iris Versicolor
- Iris Virginica

## ⚙️ Project Workflow
1. Load the dataset from `iris.csv`.
2. Remove the unnecessary `Id` column.
3. Convert species names into numerical labels.
4. Separate the input features and target variable.
5. Split the dataset into 80% training data and 20% testing data.
6. Train the Gaussian Naive Bayes classifier.
7. Evaluate the model using a confusion matrix, accuracy score, and classification report.
8. Predict the species for a custom data point.
9. Save the trained model as a Pickle file and load it again for prediction.

## 🧠 Algorithm: Gaussian Naive Bayes
Gaussian Naive Bayes is a supervised machine learning classification algorithm based on Bayes' theorem. It assumes that features follow a Gaussian (normal) distribution within each class and that features are conditionally independent given the class.

In this project, the algorithm predicts the Iris flower species using the four flower measurements.

## 📊 Model Evaluation
The model is evaluated using the following metrics:

- **Accuracy Score:** Measures the proportion of correct predictions.
- **Confusion Matrix:** Compares actual and predicted class labels.
- **Classification Report:** Displays precision, recall, F1-score, and support for each class.

The training and testing results are printed when the Python script is executed.

## 💾 Model Saving and Loading
The trained model is saved in the file `Iris_Model_Naive_Bayes.pkl` using Python's Pickle module.

The saved model is then loaded and used to predict the species for the sample input.

**Security Note:** Only load Pickle files from trusted sources.

## 📦 Installation
Install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## ▶️ How to Run the Project
1. Clone or download the repository.
2. Place `iris.csv` in the same directory as `naive_bayes.py`.
3. Open a terminal in the project directory.
4. Execute the following command:

```bash
python naive_bayes.py
```

The program displays the training and testing evaluation results, predicts the species for a custom input, and saves the trained model.

## 🚀 Future Enhancements
- Compare Gaussian Naive Bayes with other classification algorithms.
- Visualize the dataset and model performance.
- Improve preprocessing and evaluate different data splits.
- Develop a web application for interactive flower classification.

## 📝 Conclusion
This project demonstrates how Gaussian Naive Bayes can be used for multiclass classification with Python and Scikit-learn. It covers data preprocessing, model training, evaluation, prediction, and model persistence.
