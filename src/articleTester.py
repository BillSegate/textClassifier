import json
import os
import math
import time

from sklearn.naive_bayes import logsumexp
from logger import log
from articleProcessor import readFile, performPreProcessing, PATH_MEDICAL_ARTICLES, PATH_OTHER_ARTICLES


# calculate P(C)
def calculateClassProbability(className: str) -> float:
    medicalDocs = len(os.listdir(PATH_MEDICAL_ARTICLES))
    otherDocs = len(os.listdir(PATH_OTHER_ARTICLES))
    totalDocs = medicalDocs + otherDocs

    if(className == 'Medical'):
        return medicalDocs / totalDocs
    else:
        return otherDocs / totalDocs

def testArticle(article: list, medicalClassProbability: float, otherClassProbability: float, medicalFrequency: dict, otherFrequency: dict) -> None:
    totalMedSum = 0
    totalOtherSum = 0
    for word in article: 
        if word in medicalFrequency:
            probMedWord = math.log(medicalFrequency[word])
            totalMedSum = logsumexp([totalMedSum, probMedWord])
        
        if word in otherFrequency:
            probOtherWord = math.log(otherFrequency[word])
            totalOtherSum = logsumexp([totalOtherSum, probOtherWord])

    medicalProb = medicalClassProbability + totalMedSum
    otherProb = otherClassProbability + totalOtherSum
    
    if(medicalProb > otherProb):
        return 1
    else:
        return 0

def main():
    # get files in the article folder
    medicalTestArticles = os.listdir('data/test/medicalArticles')
    otherTestArticles = os.listdir('data/test/medicalArticles')
    medicalLabels = [0 for i in range(len(medicalTestArticles))]
    otherLabels = [0 for i in range(len(otherTestArticles))]
    
    testFiles = medicalTestArticles + otherTestArticles
    labels = medicalLabels + otherLabels

    log(f'starting predicting class of {len(testFiles)} files...')
    startTime = time.time()

    medicalJson = readFile('data/medicalFrequencies.json')
    medicalFrequency = json.loads(medicalJson)

    otherJson = readFile('data/otherFrequencies.json')
    otherFrequency = json.loads(otherJson)

    medicalClassProbability = calculateClassProbability('Medical')
    otherClassProbability = calculateClassProbability('Other')

    # for each file read the article
    predictedLabels = []
    for i in range(len(labels)):
        if(labels[i] == 0):
            filepath =  'data/test/medicalArticles/' + testFiles[i]
        else:
            filepath =  'data/test/otherArticles/' + testFiles[i]

        text = readFile(filepath)
        text = performPreProcessing(text)
        predictedLabels.append(testArticle(text, medicalClassProbability, otherClassProbability, medicalFrequency, otherFrequency))
    
    rightCounter = 0
    for k in range(len(predictedLabels)):
        if predictedLabels[k] == labels[k]:
            rightCounter += 1

    log(f'Finished in {round(time.time() - startTime, 3)} seconds, with an accuracy of {round((rightCounter/len(predictedLabels))*100, 3)}%')

if __name__ == '__main__':
    main()