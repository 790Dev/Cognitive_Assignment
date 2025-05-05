#Q1

import nltk
import re
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Original Paragraph
text = """
Technology has always fascinated me, especially the way it evolves and changes our daily lives.
From smartphones to artificial intelligence, every innovation opens up a world of possibilities.
I enjoy reading about the latest gadgets, understanding how they work, and imagining how they will shape the future.
The fast pace of technological advancement keeps me curious and constantly learning.
Moreover, I love discussing new trends and debating their ethical implications with friends.
"""

# 1. Lowercase and remove punctuation
text_lower = text.lower()
text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))

# 2. Tokenize
word_tokens = word_tokenize(text_no_punct)
sent_tokens = sent_tokenize(text_no_punct)

# 3. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in word_tokens if word not in stop_words]

# 4. Word Frequency
word_freq = Counter(filtered_words)
print("Word Frequencies (excluding stopwords):")
print(word_freq.most_common())









#Q2
# Initialize stemmers and lemmatizer
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Apply each
porter_stems = [porter.stem(word) for word in filtered_words]
lancaster_stems = [lancaster.stem(word) for word in filtered_words]
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]

# Display results
print("\nPorter Stemmer:", porter_stems)
print("Lancaster Stemmer:", lancaster_stems)
print("Lemmatizer:", lemmatized)





#Q3
# a. Words > 5 letters
long_words = re.findall(r'\b\w{6,}\b', text)
# b. Numbers
numbers = re.findall(r'\b\d+\b', text)
# c. Capitalized words
capitalized_words = re.findall(r'\b[A-Z][a-z]*\b', text)

# d. Only alphabet words
alpha_words = re.findall(r'\b[a-zA-Z]+\b', text)
# e. Words starting with vowels
vowel_words = [word for word in alpha_words if word.lower().startswith(('a', 'e', 'i', 'o', 'u'))]

print("\nWords > 5 letters:", long_words)
print("Numbers:", numbers)
print("Capitalized Words:", capitalized_words)
print("Alphabetic Words:", alpha_words)
print("Words starting with vowels:", vowel_words)



#Q4
# Custom tokenizer function
def custom_tokenizer(text):
    text = re.sub(r'[^\w\s\'\-\.]', '', text)  # keep hyphens, decimals, apostrophes
    tokens = re.findall(r"\d+\.\d+|\w+(?:-\w+)*|'[\w]+|\w+", text)
    return tokens

tokens_custom = custom_tokenizer(text)
print("\nCustom Tokens:", tokens_custom)

# Regex substitutions
text_cleaned = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '<EMAIL>', text)
text_cleaned = re.sub(r'http[s]?://\S+|www\.\S+', '<URL>', text_cleaned)
text_cleaned = re.sub(r'\b(\+?\d{1,3}[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{4}\b', '<PHONE>', text_cleaned)

print("\nText after substitutions:")
print(text_cleaned)


