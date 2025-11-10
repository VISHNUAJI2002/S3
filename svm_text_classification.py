'''
Program to implement text classification using Support Vector Machine
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Define the small dataset
# We will classify texts as tech(0) or finance(1)
data = [
    "Apple launched a new iPhone with better neural engine.",  # tech
    "The stock market saw huge gains after the quarterly report.", # finance
    "Google's machine learning model achieved 90% accuracy.",  # tech
    "Investors are worried about rising interest rates and inflation.", # finance
    "Python libraries like scikit-learn are great for ML.", # tech
    "Bonds and treasury yields are highly volatile this week." # finance
]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(data, [0, 1, 0, 1, 0, 1], test_size=0.2, random_state=42)
print(f"Total datapoints: {len(data)}")
print(f"Training datapoints: {len(X_train)}")
print(f"Testing datapoints: {len(X_test)}")
print("-----")

# 3. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')


# Convert training data
x_train_vectors=vectorizer.fit_transform(X_train)

# Convert testing data
x_test_vectors=vectorizer.transform(X_test)


# 4. Initialize and Train the SVM
svm_classifier = SVC(kernel='linear',C=1.0,random_state=42)
print("training svm on small dataset")
svm_classifier.fit(x_train_vectors, y_train)
print("Training complete")

# 5. Predict and Evaluate
y_pred = svm_classifier.predict(x_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)
'''print("Classification Report:")
print(classification_report(y_test, y_pred))'''


# 6. Simple Prediction
new_text = ["Artificial intelligence is transforming industries."]
new_text_vectorized = vectorizer.transform(new_text)
prediction = svm_classifier.predict(new_text_vectorized)

if prediction[0] == 0:
    print("\nPrediction: Tech")
else:
    print("\nPrediction: Finance")

