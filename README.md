# TextClassifier
The problem to solve is attributing to a text given in input (only English language) a class among two: medical/non-medical.

## Dependencies
- NLTK
- wikipedia-api
- re

## How to use
The default category will be 'Medicine' (edit articleRetriever.py in order to change it).
1. Use init.sh to clear articles folder and retrieve category articles. (optional: it will take 5 to 10 minutes to complete);
2. Use start.sh to perform operations.