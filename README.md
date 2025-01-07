# Comparing-Supervised-Machine-Learning-Models-in-Classifying-Movie-Review-Sentiment

A comparative study of different supervised machine learning models for sentiment analysis of movie reviews, implemented in Python using Jupyter Notebook.

## Overview

This project evaluates the performance of various supervised machine learning algorithms in classifying movie review sentiments. The models compared include Support Vector Machines (SVM), Logistic Regression, Naïve Bayes, K-Nearest Neighbors (KNN), Random Forest, and Artificial Neural Networks (ANN).

## Key Features

- Comprehensive text preprocessing including contraction expansion, URL removal, and lemmatization
- Implementation of multiple ML algorithms with hyperparameter tuning
- Parallel processing for improved computational efficiency
- Stratified cross-validation for robust model evaluation
- Detailed performance metrics and confusion matrices

## Dependencies

```
contractions==0.1.73
joblib==1.4.2
numpy==1.24.3
nltk==3.9.1
pandas==2.2.3
scikit-learn==1.3.2
spacy==3.7.2
jupyter notebook
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/Movie-Review-Sentiment-Analysis.git
cd Movie-Review-Sentiment-Analysis
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

4. Download the SpaCy English model
```bash
python -m spacy download en_core_web_sm
```

5. Launch Jupyter Notebook
```bash
jupyter notebook
```

## Usage

### For Movie Reviews
1. Use the default setup with the provided movie review dataset
2. Open and run the Jupyter Notebook

### For Other Sentiment Analysis Tasks
1. Prepare your dataset in CSV format with two columns:
   - 'Text': Your text data
   - 'Sentiment': Your sentiment labels
2. Place your CSV file in the `data` directory
3. Open the Jupyter Notebook and modify the filepath:
```python
# Change this line in the notebook
filepath = 'your_dataset.csv'  # Currently set to 'reduced_dataset.csv'
```
4. Run all cells in the notebook

### Dataset Requirements
- CSV format
- Must contain 'Text' and 'Sentiment' columns
- Text should be in string format
- Sentiment should be binary (0/1 or positive/negative)

## Model Performance

The comparative analysis of different models yielded the following results:

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|---------|-----------|
| SVM                 | 0.890    | 0.890     | 0.890   | 0.890     |
| Logistic Regression | 0.889    | 0.889     | 0.889   | 0.889     |
| Naïve Bayes         | 0.869    | 0.870     | 0.869   | 0.869     |
| ANN                 | 0.868    | 0.868     | 0.868   | 0.868     |
| Random Forest       | 0.858    | 0.858     | 0.858   | 0.858     |
| KNN                 | 0.759    | 0.764     | 0.759   | 0.758     |

## Features

- Text preprocessing including:
  - Contraction expansion
  - URL removal
  - Non-alphabetic character removal
  - Tokenization and lemmatization
  - Stop word removal
- TF-IDF vectorization with n-gram support
- Stratified k-fold cross-validation
- Grid search for hyperparameter optimization
- Parallel processing support
- Comprehensive evaluation metrics

## Acknowledgments

- IMDb for the movie review dataset
- All contributors and maintainers of the libraries used in this project

## Author

- [Sarthak Shakya]
