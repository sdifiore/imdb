import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

directory = "d:/imdb/Data"
filename = "IMDB Dataset.csv"

file_path = os.path.join(directory, filename)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())

# Function to remove HTML tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Apply the function to the 'review' column
df['clean_review'] = df['review'].apply(remove_html)

# Display the cleaned reviews
print(df[['review', 'clean_review']].head())

# Map 'positive' to 1 and 'negative' to 0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Verify the mapping
print(df[['sentiment', 'label']].head())
# Features and target variable
X = df['clean_review']
y = df['label']
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Features and target variable
X = df['clean_review']
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the vectorizer
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
X_train_vect = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_vect = vectorizer.transform(X_test)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_vect, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vect)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
