from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from logger import log
import os
import re
import json

# nltk stuff
stoplist = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# path to articles
PATH_MEDICAL_ARTICLES = 'data/medicalArticles'
PATH_OTHER_ARTICLES = 'data/otherArticles'

# perform the stopwords elimination
def deleteStopwords(article: str) -> str:
    return [word.lower() for word in article.split() if word.lower() not in stoplist]

def removeNonAlphanumeric(article: str) -> str:
    # whenever a non alphanumeric character is found, replace it with a space
    # exception: apostrophes and spaces
    return re.sub(r'[^a-zA-Z0-9\s\']', ' ', article)

# perform the lemmatization of the article
def lemmatizeArticle(article: list) -> list:
    return [lemmatizer.lemmatize(word) for word in article]


# count document occurrences
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

# create the Bag of Words based on document occurrences
def createBoW(isMedical: bool) -> None:
    # different path if isMedical or not
    if isMedical:
        pathToArticles = PATH_MEDICAL_ARTICLES
        jsonName = 'data/medicalOccurrences.json'
    else:
        pathToArticles = PATH_OTHER_ARTICLES
        jsonName = 'data/otherOccurrences.json'

    # get files in the article folder
    articlesDirectory = os.listdir(pathToArticles)
    occurrencesDict = {}
    log(f'{len(articlesDirectory)} articles found in {pathToArticles}')
    # for each file read the article
    for filename in articlesDirectory:
        filepath =  pathToArticles + '/' + filename
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # clear the text 
            text = ''
            for line in lines:
                text += line.strip() + '\n'
            # process the text
            text = performPreProcessing(text)
            # count occurrences, we use set in order to have unique words.
            # this is useful if we want to calculate the document occurrences
            countOccurrences(occurrencesDict, set(text))
    # end for

    # sort the dict in order to have word with high occurrence at top
    occurrencesDict = dict(sorted(occurrencesDict.items(), key=lambda item: item[1], reverse=True))
    
    # save dict in a json file
    with open(jsonName, 'w', encoding='utf-8') as jsonFile:
        json.dump(occurrencesDict, jsonFile, indent=2)

def main():
    createBoW(True)
    createBoW(False)


if __name__ == '__main__':
    main()
