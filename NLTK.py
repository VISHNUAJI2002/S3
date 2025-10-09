import nltk
from nltk.corpus import stopwords

nltk.download('all')

#1) perform word and sentence tokenization.
from nltk import word_tokenize, sent_tokenize, bigrams

text="This is a sample text used. For testing purpose only Happiness"
print("Word Tokenization and Sentence tokenization:")
print(word_tokenize(text))
print(sent_tokenize(text))

print()

#2) Remove the stop words from the given text
word="The sun was shining brightly, and we decided to go for a walk in the park."
stop_words=set(stopwords.words('english'))
tokens=word_tokenize(word.lower())

print("Original text:",word)
print("Stop word removal:")
filtered_tokens=[word for word in tokens if word not in stop_words]
print(filtered_tokens)

print()

#3) Perform Part of Speech tagging
from nltk import pos_tag
from nltk import sent_tokenize

print("Part of speech:")
tokenized_text=word_tokenize(text)
print (pos_tag(tokenized_text))

print()

#4) create n-grams for different values of n=2,4.
print("n-grams 2&4:")
from nltk import ngrams
bigrams_2 = ngrams(tokenized_text,2)
bigrams_4 = ngrams(tokenized_text,4)
print(list(bigrams_2))
print(list(bigrams_4))
