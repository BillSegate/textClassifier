#!/bin/sh

# train articles
if test "$(ls data/train/medicalArticles/)" != ""; then rm data/train/medicalArticles/*; fi
if test "$(ls data/train/otherArticles/)" != ""; then rm data/train/otherArticles/*; fi
# test articles
if test "$(ls data/test/medicalArticles/)" != ""; then rm data/test/medicalArticles/*; fi
if test "$(ls data/test/otherArticles/)" != ""; then rm data/test/otherArticles/*; fi

# execute script
python src/articleRetriever.py

exit 1