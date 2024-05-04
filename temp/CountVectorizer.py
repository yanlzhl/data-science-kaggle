from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample text data
documents = ["This is the first document",
             "This document is about machine learning of this",
             "Another example of a document"]

# Create CountVectorizer object
vectorizer = CountVectorizer(min_df=0.0,max_df=1.0,binary=False,ngram_range=(1,3))

# Fit and transform the data (learn vocabulary and create matrix)
X = vectorizer.fit_transform(documents)

# Convert the sparse matrix to a dense numpy array (optional)
X_dense = X.toarray()

# Create a pandas DataFrame from the dense array and vocabulary
df = pd.DataFrame(X_dense, columns=vectorizer.get_feature_names_out())

print(df)