#!/bin/sh

# train articles
rm /data/train/medicalArticles/*
rm /data/train/otherArticles/*
# test articles
rm /data/test/medicalArticles/*
rm /data/test/otherArticles/*

# execute script
python src/articleRetriever.py

exit 1