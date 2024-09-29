## Problem Statement

The aim of this project is to analyze and classify sentiment in text data as either positive or negative. This involves comparing traditional machine learning methods like Logistic Regression and SVM, using natural language processing tools such as NLTK, with modern transformer-based models like DistilBERT. The project focuses on evaluating the accuracy and performance of these approaches in sentiment analysis tasks.

## Installations

To install the necessary packages, run the following command:
```
pip install random nltk torch scikit-learn transformers
```

## How to Run

Ensure that all three files are located in the same directory.

### Troubleshooting NLTK Data Download

If you encounter issues downloading NLTK data, you can resolve this by either of the following methods:

1. Add the following lines at the beginning of your code:
```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

2. Run the following command in your terminal to download the required data:
```
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```
