#!/bin/sh

# execute script
python src/articleProcessor.py
printf "\n== TEST == \n\n"
python src/articleTester.py 