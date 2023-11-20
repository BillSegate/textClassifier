#!/bin/sh

# execute script
python src/articleProcessor.py &
wait
python src/articleTester.py 