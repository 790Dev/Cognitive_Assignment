#q1
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Sample paragraph (about technology)
paragraph = """
Technology fascinates me endlessly. From AI assistants to space exploration, every breakthrough feels like magic.
I often read tech blogs, watch gadget reviews, and follow news about innovations.
It’s amazing how quickly things change and how deeply technology affects our lives.
Discussing future possibilities with friends keeps me excited.
"""

# 1. Lowercase and remove punctuation using re
text_clean = re.sub(r'[^\w\s]', '', paragraph.lower())

# 2. Tokenization into words and sentences
word_tokens = word_tokenize(text_clean)
sent_tokens = sent_tokenize(text_clean)

# 3. Comparison between split() and word_tokenize()
split_words = text_clean.split()
print("Python split():", split_words)
print("\nNLTK word_tokenize():", word_tokens)

# 4. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in word_tokens if word not in stop_words]

# 5. Word Frequency Distribution
word_freq = Counter(filtered_words)
print("\nWord Frequency (excluding stopwords):")
print(word_freq.most_common())




#Q2
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Downloads
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Extract all words with only alphabets
alpha_words = re.findall(r'\b[a-zA-Z]+\b', text_clean)

# 2. Remove stopwords again
filtered_alpha = [word for word in alpha_words if word not in stop_words]

# 3. Stemming
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in filtered_alpha]

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_alpha]

# 5. Compare outputs
print("Original Words:", filtered_alpha)
print("\nStemmed Words:", stemmed)
print("\nLemmatized Words:", lemmatized)

# ➤ Explanation:
print("""
Stemming often cuts words to their base/root form without considering grammar,
e.g., 'technologies' → 'technolog'.

Lemmatization is more accurate because it uses vocabulary and grammar rules,
e.g., 'technologies' → 'technology'.

Prefer lemmatization when you need meaningful words (e.g., for readability or semantic analysis).
Use stemming when performance matters more than precision (e.g., search indexing).
""")




#Q3
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# Three short texts
texts = [
    "Apple releases new iPhone with better battery and camera.",
    "The movie received mixed reviews from critics.",
    "This laptop is lightweight and perfect for travel."
]

# 1. CountVectorizer (Bag of Words)
cv = CountVectorizer()
bow_matrix = cv.fit_transform(texts)
print("Bag of Words:\n", bow_matrix.toarray())
print("Vocabulary:\n", cv.get_feature_names_out())

# 2. TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts)
print("\nTF-IDF Matrix:\n", tfidf_matrix.toarray())
print("TF-IDF Features:\n", tfidf.get_feature_names_out())

# 3. Top 3 keywords from each text using TF-IDF
for i, row in enumerate(tfidf_matrix.toarray()):
    top_indices = row.argsort()[::-1][:3]
    top_keywords = [tfidf.get_feature_names_out()[j] for j in top_indices]
    print(f"\nTop 3 keywords for Text {i+1}:", top_keywords)





#Q4
from sklearn.metrics.pairwise import cosine_similarity

# Two short texts (technology comparison)
text1 = """
Artificial Intelligence is transforming industries by automating tasks and learning from data.
It powers chatbots, recommendation systems, and self-driving cars.
"""

text2 = """
Blockchain is a decentralized ledger that ensures transparency and immutability of records.
It is used in cryptocurrencies, supply chains, and secure transactions.
"""

# Preprocess and tokenize
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

tokens1 = preprocess(text1)
tokens2 = preprocess(text2)

# a. Jaccard Similarity
set1, set2 = set(tokens1), set(tokens2)
jaccard = len(set1 & set2) / len(set1 | set2)
print("Jaccard Similarity:", round(jaccard, 4))

# b. Cosine Similarity with TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])
cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print("Cosine Similarity:", round(float(cos_sim), 4))

# c. Analysis
print("""
Cosine similarity captures deeper context and weighting of words (e.g., importance of terms).
Jaccard is simpler and only based on overlap.
In this case, cosine similarity gives better insight into semantic similarity of topics.
""")




#Q5
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample reviews
reviews = [
    "The camera quality is amazing and the battery lasts long.",
    "Terrible service. My issue was never resolved.",
    "Packaging was okay, but delivery was late.",
    "I love the features and sleek design of this product.",
    "Not worth the money. I'm disappointed."
]

# Analyze sentiment using TextBlob
for review in reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
    print(f"\nReview: {review}")
    print(f"Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}, Sentiment: {sentiment}")

# Word Cloud for positive reviews
positive_texts = " ".join([review for review in reviews if TextBlob(review).sentiment.polarity > 0.1])
wordcloud = WordCloud(width=600, height=300, background_color='white').generate(positive_texts)

# Show word cloud
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Positive Reviews")
plt.show()





#q6
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample reviews
reviews = [
    "The camera quality is amazing and the battery lasts long.",
    "Terrible service. My issue was never resolved.",
    "Packaging was okay, but delivery was late.",
    "I love the features and sleek design of this product.",
    "Not worth the money. I'm disappointed."
]

# Analyze sentiment using TextBlob
for review in reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
    print(f"\nReview: {review}")
    print(f"Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}, Sentiment: {sentiment}")

# Word Cloud for positive reviews
positive_texts = " ".join([review for review in reviews if TextBlob(review).sentiment.polarity > 0.1])
wordcloud = WordCloud(width=600, height=300, background_color='white').generate(positive_texts)

# Show word cloud
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Positive Reviews")
plt.show()




