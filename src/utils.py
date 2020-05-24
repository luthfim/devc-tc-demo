import re
from bs4 import BeautifulSoup
import nltk

def print_sample(df, message, n=10, col='content'):
    print(message)
    print(df[col].iloc[:n])
    print()

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_special_characters(text):
    pattern = r'[^\w]'
    text = re.sub(pattern, ' ',text)
    return text

def preprocessing_pipeline(text):
    text = strip_html(text)
    text = remove_special_characters(text)
    return text

ps = nltk.porter.PorterStemmer()
def simple_stemmer(text):
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=True):
    tokens = text.split()
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
