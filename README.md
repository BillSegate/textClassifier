# TextClassifier
The problem to solve is attributing to a text given in input (only English language) a class among two: medical/non-medical.

## Dependencies
- NLTK
- wikipedia-api
- re

## How to use
1. Use init.sh to clear articles folder and retrieve category articles;
2. Use start.sh to perform operations.

## Theory
After retrieving all articles, for each article execute some operations in order to have texts pre-processed:
1. articles must be cleaned from non-alphanumeric character (excluding the apostrophes);
2. perform the ***stopwords*** elimination; 
3. perform ***lemmatization***.

After the pre-processing phase, we calculate the ***Bag of Words (BoW)*** for both medical articles and other categories articles. Moreover, we calculate the document frequencies of each word for both BoWs.

To classify a given article, we use the ***Naive Bayes classifier***.