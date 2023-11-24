import wikipediaapi
import re
import time
import random
import math
from logger import log


# init wikipedia API
userAgent = 'textClassifier (NLP project)'
WIKI = wikipediaapi.Wikipedia(language='en', user_agent=userAgent)
# categories
MEDICAL_CATEGORY = ['Pathology', 'Pediatrics', 'Neurology', 'Cardiology', 'Oncology']
OTHER_CATEGORIES = ['Politics', 'Ecology', 'Computer security', 'Electricity', 'Trigonometry']


# Utils function
def makeValidFilename(title: str) -> str:
    # Replace invalid characters with underscores
    validChars = re.sub(r'[^\w\s.-]', '_', title)
    # Remove leading and trailing whitespaces
    validChars = validChars.strip()
    return validChars

# given the category members retrieve titles from leaves in the category tree
def retrieveTitles(categoryMembers: dict, level: int=0, max_level: int=1) -> list:
    titles = []
    for c in categoryMembers.values():
        # if there is a subcategory and it can go lower in the category tree.
        # each category could have n sub categories, but since there are a plethora of pages we reduce the search in order to save time
        if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            titles.extend(retrieveTitles(c.categorymembers, level=level+1, max_level=max_level))
        # else if the category member is an article (check on Namespace.MAIN), it appends the title article.
        elif c.ns == wikipediaapi.Namespace.MAIN:
            titles.append(c.title)
    return titles

# given a list of titles retrieve each articles
def retrieveArticles(titles: list, isMedical: bool) -> None:
    numTitle = len(titles)
    trainLen = math.ceil(numTitle / 100 * 80)
    k = 1
    for title in titles:
        # retrieve the article based on title
        article = WIKI.page(title)
        # since the title could have non valid chars, remove them
        filename = makeValidFilename(title)

        if trainLen > k:
            trainTestFolder = 'data/train/'
        else:
            trainTestFolder = 'data/test/'

        # different path for medical articles
        if isMedical:
            folder = 'medicalArticles/'
        else:
            folder = '/otherArticles/'

        filepath = trainTestFolder + folder + filename + '.txt'

        # save the article in a file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.writelines(article.text)

        if(k % 25 == 0 or k == len(titles)):
            log(f'loading {k}/{numTitle} articles...')
        k += 1

# given a category it will retrieve all documents.
# it won't search on subcategories for time and space saving.
# in order to download articles from subcategories, just edit the MAX_SUBCATEGORIES constant.
def retrieve(category: str, isMedical: bool) -> None:
    MAX_SUBCATEGORIES = 0
    startTime = time.time()
    log(f'Retrieving category: {category}')
    # retrieve primary category
    categoryResult = WIKI.page('Category:' + category)
    # retrieve all titles
    titles = retrieveTitles(categoryResult.categorymembers, 0, MAX_SUBCATEGORIES)
    
    # shuffles the list of titles in order to subdivide it in 80% train and 20% test
    random.shuffle(titles)

    log(f'articles found: {len(titles)}')
    # retrieve all articles
    retrieveArticles(titles, isMedical)
    
    log(f'{category} retrieved in {round(time.time() - startTime, 2)}s.')

def main():
        # first retrieve medical categories
    for category in MEDICAL_CATEGORY:
        retrieve(category, True)
        
    # then retrieve other categories
    for category in OTHER_CATEGORIES:
        retrieve(category, False)

if __name__ == "__main__":
    main()