from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import re
import json

stoplist = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def deleteStopwords(article: str) -> str:
    return [word.lower() for word in article.split() if word.lower() not in stoplist]

def removeNonAlphanumeric(article: str) -> str:
    # whenever a non alphanumeric character is found, replace it with a space
    # exception: apostrophes and spaces
    return re.sub(r'[^a-zA-Z0-9\s\']', ' ', article)

def lemmatizeArticle(article: list) -> list:
    return [lemmatizer.lemmatize(word) for word in article]


# count occurrences
def countOccurrences(occurrencesDict: dict, article: set) -> None:
    for word in article:
        if word in occurrencesDict:
            occurrencesDict[word] += 1
        else:
            occurrencesDict[word] = 1

def performPreProcessing(article: str) -> list:
    # remove non alphanumeric 
    text = removeNonAlphanumeric(article)
    # perform the stopwords elimination
    text = deleteStopwords(text)
    # perform lemmatization
    text = lemmatizeArticle(text)

    return text

def main():
    # get files in the article folder
    articlesDirectory = os.listdir('articles')
    occurrencesDict = {}
    # for each file read the article
    for filename in articlesDirectory:
        filepath = 'articles/' + filename
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            text = ''
            for line in lines:
                text += line.strip() + '\n'
            # process the text
            text = performPreProcessing(text)
            # count occurrences
            countOccurrences(occurrencesDict, set(text))

    occurrencesDict = dict(sorted(occurrencesDict.items(), key=lambda item: item[1], reverse=True))

    with open('occurrences.json', 'w', encoding='utf-8') as json_file:
        json.dump(occurrencesDict, json_file, indent=2)


if __name__ == '__main__':
    main()
