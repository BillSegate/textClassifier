import wikipediaapi
import re
import time
from logger import log

# init wikipedia API
userAgent = 'textClassifier (NLP project)'
WIKI = wikipediaapi.Wikipedia(language='en', user_agent=userAgent)
# categories
MEDICAL_CATEGORY = 'Medicine'
OTHER_CATEGORIES = ['Physics', 'Mathematics', 'Human impact on the environment']

# Utils function
def makeValidFilename(title: str) -> str:
    # Replace invalid characters with underscores
    validChars = re.sub(r'[^\w\s.-]', '_', title)
    # Remove leading and trailing whitespaces
    validChars = validChars.strip()
    return validChars

# given the category members retrieve titles from leaves in the category tree
def retrieveTitles(categoryMembers, level=0, max_level=1):
    titles = []
    for c in categoryMembers.values():
        # if there is a subcategory and it can go lower in the category tree.
        # each category could have n sub categories, but since there are a plethora of pages we reduce the search in order to save time
        # retrieve just one subcategory level takes up to 10 minutes and it retrieves 1522 articles (medical articles).
        if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            titles.extend(retrieveTitles(c.categorymembers, level=level+1, max_level=max_level))
        # else if the category member is an article (check on Namespace.MAIN), it appends the title article.
        elif c.ns == wikipediaapi.Namespace.MAIN:
            titles.append(c.title)
    return titles

# given a list of titles retrieve each articles
def retrieveArticles(titles: list, isMedical: bool) -> None:
    numTitle = len(titles)
    k = 1
    for title in titles:
        # retrieve the article based on title
        article = WIKI.page(title)
        # since the title could have non valid chars, remove them
        filename = makeValidFilename(title)

        # different path for medical articles
        if isMedical:
            filepath = 'data/medicalArticles/'
        else:
            filepath = 'data/otherArticles/'

        filepath = filepath + filename + '.txt'

        # save the article in a file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.writelines(article.text)

        log(f'loading {k}/{numTitle} articles...')
        k += 1

def retrieve(category: str, isMedical: bool) -> None:
    start_time = time.time()
    log(f'Retrieving category: {category}')
    # retrieve primary category
    categoryResult = WIKI.page('Category:' + category)
    # retrieve all titles
    titles = retrieveTitles(categoryResult.categorymembers)

    log(f'articles found: {len(titles)}')
    # retrieve all articles
    retrieveArticles(titles, isMedical)
    
    log(f'{category} retrieved in {time.time() - start_time} seconds!')

def main():
    # first retrieve medical articles
    retrieve(MEDICAL_CATEGORY, True)
    # then retrieve other categories
    for category in OTHER_CATEGORIES:
        retrieve(category, False)

if __name__ == "__main__":
    main()