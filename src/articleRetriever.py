import wikipediaapi
import re

userAgent = 'textClassifier (marco.santi_02@studenti.univr.it)'
WIKI = wikipediaapi.Wikipedia(language='en', user_agent=userAgent)
# 
SELECTED_CATEGORY = 'Medicine'

# Utils function
def makeValidFilename(title):
    # Replace invalid characters with underscores
    validChars = re.sub(r'[^\w\s.-]', '_', title)
    # Remove leading and trailing whitespaces
    validChars = validChars.strip()
    return validChars

# given the category members retrieve titles from leaves in the category tree
def retrieveTitles(category_members, level=0, max_level=1):
    titles = []
    for c in category_members.values():
        if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            titles.extend(retrieveTitles(c.categorymembers, level=level+1, max_level=max_level))
        elif c.ns == wikipediaapi.Namespace.MAIN:
            titles.append(c.title)
    return titles

# given a list of titles retrieve each articles
def retrieveArticles(titles):
    for title in titles:
        article = WIKI.page(title)
        filename = makeValidFilename(title)
        filePath = 'articles/' + filename + '.txt'
        with open(filePath, 'w', encoding='utf-8') as file:
            file.writelines(article.text)

# retrieve first node of the selected category
categoryResult = WIKI.page('Category:' + SELECTED_CATEGORY)
# retrieve all titles
titles = retrieveTitles(categoryResult.categorymembers)
# retrieve all articles
retrieveArticles(titles)