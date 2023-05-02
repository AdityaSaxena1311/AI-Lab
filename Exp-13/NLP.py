import nltk
nltk.data.path.append("/path/to/your/nltk_data")
from nltk.book import *

# Print the available texts and sentences
print("Available texts:", text1, text2, text3, text4, text5, text6, text7, text8, text9)
print("Available sentences:", sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent8, sent9)

# Print the tokens and length of sent7
print("sent7 tokens:", sent7)
print("sent7 length:", len(sent7))

# Print the unique tokens in text7
print("Unique tokens in text7:", list(set(text7))[:10])

# Print the frequency of words in text7
dist = FreqDist(text7)
print("Number of unique words in text7:", len(dist))
vocab1 = list(dist.keys())
print("First 10 words in vocab1:", vocab1[:10])
freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
print("Frequent words in text7 with length > 5 and frequency > 100:", freqwords)

# Stemming using Porter Stemmer
input1 = 'List listed lists listing listings'
words1 = input1.lower().split(' ')
porter = nltk.PorterStemmer()
stemmed_words1 = [porter.stem(t) for t in words1]
print("Stemmed words using Porter Stemmer:", stemmed_words1)

# Tokenization and sentence splitting
text11 = "Children shouldn't drink a sugary drink before bed."
print("Tokenized text11 using split():", text11.split(' '))
print("Tokenized text11 using word_tokenize():", nltk.word_tokenize(text11))
text12 = 'This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!'
sentences = nltk.sent_tokenize(text12)
print("Sentences in text12 using sent_tokenize():", sentences)
