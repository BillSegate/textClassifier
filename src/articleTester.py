import json
import os
import math
import time
import numpy as np

from logger import log
from articleProcessor import readFile, performPreProcessing, PATH_MEDICAL_ARTICLES, PATH_OTHER_ARTICLES


# calculate P(C) = |docsⱼ| / |total docs|
def calculateClassProbability(className: str) -> float:
    medicalDocs = len(os.listdir(PATH_MEDICAL_ARTICLES))
    otherDocs = len(os.listdir(PATH_OTHER_ARTICLES))
    totalDocs = medicalDocs + otherDocs

    if(className == 'Medical'):
        return medicalDocs / totalDocs
    else:
        return otherDocs / totalDocs

# classify a given article, having the dictionary with the frequencies, and the class probabilities
# Remember that: values inside the frequencies dictionary are already the log of P(Wₖ| Cⱼ)
def testArticle(article: list, medicalClassProbability: float, otherClassProbability: float, medicalFrequency: dict, otherFrequency: dict) -> None:
    totalMedSum = 0
    totalOtherSum = 0
    for word in article: 
        if word in medicalFrequency:
            probMedWord = medicalFrequency[word]
            totalMedSum += probMedWord
        
        if word in otherFrequency:
            probOtherWord = otherFrequency[word]
            totalOtherSum += probOtherWord

    medicalClassProbability = math.log(medicalClassProbability)
    otherClassProbability = math.log(otherClassProbability)

    # in order to have all positive values we do the absolute of the sum of both class probability
    transformPositive = abs(medicalClassProbability + otherClassProbability)
    # We sum the previous value to the class probability, we are preserving the importance of the class
    medicalProb = (medicalClassProbability + transformPositive) + totalMedSum
    otherProb = (otherClassProbability + transformPositive) + totalOtherSum
    
    # return medical or non medical based on probability just calculated
    if(medicalProb > otherProb):
        return 0
    else:
        return 1

def main():
    # get files in the article folder
    medicalTestArticles = os.listdir('data/test/medicalArticles')
    otherTestArticles = os.listdir('data/test/otherArticles')
    medicalLabels = [0 for i in range(len(medicalTestArticles))]
    otherLabels = [1 for i in range(len(otherTestArticles))]
    
    # concatenate arrays
    testFiles = medicalTestArticles + otherTestArticles
    labels = medicalLabels + otherLabels

    log(f'starting predicting class of {len(testFiles)} files...')
    startTime = time.time()

    # read the frequencies files
    medicalJson = readFile('data/medicalFrequencies.json')
    medicalFrequency = json.loads(medicalJson)

    otherJson = readFile('data/otherFrequencies.json')
    otherFrequency = json.loads(otherJson)

    # get the class probabilities
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
    
    # create the confusion matrix
    confusionMatrix = np.zeros((2, 2))
    for k in range(len(predictedLabels)):
        confusionMatrix[predictedLabels[k], labels[k]] += 1

    # visualize results
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
    # calculate and visualize the statistics
    recall = truePositive/(truePositive + falseNegative)
    precision = truePositive/(truePositive + falsePositive)
    accuracy = wellPredicted/len(labels)
    fMeasure = (2*(recall*precision)) / (recall + precision)

    log(f'\nRecall: {round(recall*100, 3)}%')
    log(f'Precision: {round(precision*100, 3)}%')
    log(f'F-Measure: {round(fMeasure*100, 3)}%\n\n')
    log(f'Accuracy: {round(accuracy*100, 3)}%\n')

if __name__ == '__main__':
    main()