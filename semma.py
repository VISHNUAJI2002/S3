import nltk
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer

from nltk import sent_tokenize,word_tokenize
text="This is a sample text used. For testing purpose only"
print(word_tokenize(text))
print(sent_tokenize(text))

semma = PorterStemmer()
print(semma.stem("Happiness"))

lemma = WordNetLemmatizer()
print(lemma.lemmatize('Happiness','a'))