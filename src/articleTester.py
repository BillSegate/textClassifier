import json
import os
import math
import time
import numpy as np

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
            totalMedSum -= probMedWord
        
        if word in otherFrequency:
            probOtherWord = math.log(otherFrequency[word])
            totalOtherSum -= probOtherWord

    medicalProb = math.log(medicalClassProbability) + totalMedSum
    otherProb = math.log(otherClassProbability) + totalOtherSum
    
    if(medicalProb > otherProb):
        return 0
    else:
        return 1

def excludeCommonWords(dict1: dict, dict2: dict) -> dict:
    newDict = {}
    for (word, occurrence) in dict1.items():
        if word not in dict2:
            newDict[word] = occurrence
    return newDict


def main():
    # get files in the article folder
    medicalTestArticles = os.listdir('data/test/medicalArticles')
    otherTestArticles = os.listdir('data/test/otherArticles')
    medicalLabels = [0 for i in range(len(medicalTestArticles))]
    otherLabels = [1 for i in range(len(otherTestArticles))]
    
    testFiles = medicalTestArticles + otherTestArticles
    labels = medicalLabels + otherLabels

    log(f'starting predicting class of {len(testFiles)} files...')
    startTime = time.time()

    medicalJson = readFile('data/medicalFrequencies.json')
    medicalFrequency = json.loads(medicalJson)

    otherJson = readFile('data/otherFrequencies.json')
    otherFrequency = json.loads(otherJson)
    
    newMedicalFrequency = excludeCommonWords(medicalFrequency, otherFrequency)
    newOtherFrequency = excludeCommonWords(otherFrequency, medicalFrequency)

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
        predictedLabels.append(testArticle(text, medicalClassProbability, otherClassProbability, newMedicalFrequency, newOtherFrequency))
    
    #rightCounter = 0
    confusionMatrix = np.zeros((2, 2))
    for k in range(len(predictedLabels)):
        confusionMatrix[predictedLabels[k], labels[k]] += 1

    log(f'Finished in {round(time.time() - startTime, 3)}s.')
    log('\n == SOME STATISTICS == \n')
    log('PREDICTED/ACTUAL LABELS\t\tMEDICAL\t\tNON MEDICAL')

    for row in range(confusionMatrix.shape[0]):
        printRow = ""
        if(row == 0): 
            printRow += 'MEDICAL\t\t\t\t'
        else:
            printRow += 'NON MEDICAL\t\t\t'
        for column in range(confusionMatrix.shape[1]):
            printRow += str(int(confusionMatrix[row, column])) + '\t\t'
        log(printRow)
    
    truePositive = int(confusionMatrix[0,0])
    trueNegative = int(confusionMatrix[1,1])
    falsePositive = int(confusionMatrix[0,1])
    falseNegative = int(confusionMatrix[1,0])

    wellPredicted = truePositive + trueNegative
    log(f'\nRecall: {round((truePositive/(truePositive + falseNegative))*100, 3)}%')
    log(f'Precision: {round((truePositive/(truePositive + falsePositive))*100, 3)}%')
    log(f'Accuracy: {round((wellPredicted/len(labels))*100, 3)}%\n')

if __name__ == '__main__':
    main()