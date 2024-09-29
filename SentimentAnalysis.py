import random
import nltk
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from nltk import pos_tag
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

def load_data(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        return file.readlines()

positive_data = load_data("rt-polarity.pos")
negative_data = load_data("rt-polarity.neg")

# Training set: First 4000 positive and negative samples
X_train = positive_data[:4000] + negative_data[:4000]
y_train = [1] * 4000 + [0] * 4000

# Validation set: Next 500 samples
X_val = positive_data[4000:4500] + negative_data[4000:4500]
y_val = [1] * 500 + [0] * 500

# Test set: Remaining samples 831
X_test = positive_data[4500:] + negative_data[4500:]
y_test = [1] * 831 + [0] * 831


# Shuffle validation and test data for better training results
def shuffle_data(X_data, y_data):
    combined_data = list(zip(X_data, y_data))
    random.shuffle(combined_data)
    return zip(*combined_data)

X_val, y_val = shuffle_data(X_val, y_val)
X_test, y_test = shuffle_data(X_test, y_test)

# Tokenize sentences using NLTK's word_tokenize
def tokenize_sentences(sentences):
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_sentences

tokenized_train = tokenize_sentences(X_train)


# Extract sentiment words and their counts based on POS tagging
def extract_sentiment_words_with_counts(sentences, labels):
    sentiment_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})

    for sentence, label in zip(sentences, labels):
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)

        for word, tag in pos_tags:
            if tag.startswith('JJ') or tag.startswith('RB'):
                if label == 1:  # Positive sentiment
                    sentiment_counts[word.lower()]['positive'] += 1
                elif label == 0:  # Negative sentiment
                    sentiment_counts[word.lower()]['negative'] += 1

    sentiment_dictionary = {}
    for word, counts in sentiment_counts.items():
        if counts['positive'] > 0 and counts['negative'] == 0:
            sentiment_dictionary[word] = 1
        elif counts['negative'] > 0 and counts['positive'] == 0:
            sentiment_dictionary[word] = -1
        elif counts['positive'] > counts['negative']:
            sentiment_dictionary[word] = 1
        elif counts['negative'] > counts['positive']:
            sentiment_dictionary[word] = -1

    return sentiment_dictionary

sentiment_dict= extract_sentiment_words_with_counts(X_train, y_train)

# Sentiment prediction using dictionary approach
def predict_sentiment(sentences, sentiment_dict):
    predicted_labels = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        score = sum(sentiment_dict.get(token.lower(), 0) for token in tokens)
        predicted_labels.append(1 if score > 0 else 0)
    return predicted_labels

predicted_labels = predict_sentiment(X_test, sentiment_dict)


# Evaluate model using sklearn's metrics
print(classification_report(y_test, predicted_labels))
print(f"Accuracy: {accuracy_score(y_test, predicted_labels):.2f}")

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Extract TP, TN, FP, FN from the confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate precision, recall, and F1-score
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print the confusion matrix values and scores
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Print the full classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

# Print accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")



"""## Using Naive Bayes"""


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict the sentiment labels for the validation set
y_val_pred = nb_classifier.predict(X_val_tfidf)

# Predict the sentiment labels for the test set
y_test_pred = nb_classifier.predict(X_test_tfidf)


# Evaluate on validation set
print("Validation Set Metrics:")
print(classification_report(y_val, y_val_pred))
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")

# Evaluate on test set
print("Test Set Metrics:")
print(classification_report(y_test, y_test_pred))
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")

# Try different values for alpha (smoothing parameter)
nb_classifier = MultinomialNB(alpha=0.01)
nb_classifier.fit(X_train_tfidf, y_train)

# Predict and evaluate again
y_test_pred = nb_classifier.predict(X_test_tfidf)
print(f"Test Accuracy with alpha=0.1: {accuracy_score(y_test, y_test_pred):.7f}")

# Get confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Extract TP, TN, FP, FN
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate precision, recall, and F1-score
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print confusion matrix values and other metrics
print(f"\nConfusion Matrix:\n{conf_matrix}")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")


"""## Using lemmatisation and handling negation:"""

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define negation words
negation_words = {"not", "no", "never", "n't", "cannot"}

def preprocess_sentence(sentence, window=3):
    tokens = word_tokenize(sentence)
    lemmatized_tokens = []
    negating = False
    negation_count = 0

    for token in tokens:
        token = lemmatizer.lemmatize(token.lower())
        if token in negation_words:
            negating = True
            negation_count = 0
        elif negating and negation_count < window:
            token += "_not"
            negation_count += 1
        lemmatized_tokens.append(token)

    return " ".join(lemmatized_tokens)


X_train_preprocessed = [preprocess_sentence(sentence, window=5) for sentence in X_train]
X_val_preprocessed = [preprocess_sentence(sentence, window=5) for sentence in X_val]
X_test_preprocessed = [preprocess_sentence(sentence, window=5) for sentence in X_test]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train_preprocessed, y_train)

# Predict on validation set
val_predictions = model.predict(X_val_preprocessed)
print("Validation Classification Report:")
print(classification_report(y_val, val_predictions))

# Predict on test set
test_predictions = model.predict(X_test_preprocessed)
print("Test Classification Report:")
print(classification_report(y_test, test_predictions))

# Calculate accuracy on test set
accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Set Accuracy: {accuracy:.2f}")

# Calculate Confusion Matrix, TP, TN, FP, FN, Precision, Recall, and F1-Score for the test set
conf_matrix = confusion_matrix(y_test, test_predictions)

# Extract TP, TN, FP, FN
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate precision, recall, and F1-score
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print confusion matrix and calculated metrics
print(f"\nConfusion Matrix:\n{conf_matrix}")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")



"""## BERT Model:"""

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Move model to CPU
device = torch.device("cpu")
model.to(device)

# Preprocessing function for DistilBERT
def preprocess_bert(sentences, tokenizer, max_length=128):
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

# Preprocess the training data
X_train_encodings, X_train_attention_masks = preprocess_bert(X_train, tokenizer)

# Convert to tensors
X_train_encodings = X_train_encodings.to(device)
X_train_attention_masks = X_train_attention_masks.to(device)
y_train_tensor = torch.tensor(y_train).to(device)

# Prepare DataLoader for batch processing
batch_size = 16
train_data = TensorDataset(X_train_encodings, X_train_attention_masks, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Model training loop
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_masks, labels = batch

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} finished. Loss: {loss.item()}")

X_val_encodings, X_val_attention_masks = preprocess_bert(X_val, tokenizer)

with torch.no_grad():
    model.eval()
    X_val_encodings = X_val_encodings.to(device)
    X_val_attention_masks = X_val_attention_masks.to(device)
    outputs = model(X_val_encodings, attention_mask=X_val_attention_masks)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# Print predictions and evaluate
print("Predictions:", predictions)
print(classification_report(y_val, predictions))

# Preprocess the test data
X_test_encodings, X_test_attention_masks = preprocess_bert(X_test, tokenizer)

with torch.no_grad():
    model.eval()
    X_test_encodings = X_test_encodings.to(device)
    X_test_attention_masks = X_test_attention_masks.to(device)
    outputs = model(X_test_encodings, attention_mask=X_test_attention_masks)
    test_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# Print test predictions and evaluate
print("Test Predictions:", test_predictions)
print(classification_report(y_test, test_predictions))

