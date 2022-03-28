import re

import nltk
from cleantext import clean

# Recoge todas las "stopwords" del inglés
STOPWORDS = nltk.corpus.stopwords.words('english')

REPLACE_SPACES = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')


# elimina simbolos, espacios y stopwords
def clean_text(text):
    '''
        text: a string
        return: modified initial string
    '''

    text = text.lower()  # minusculas
    text = REPLACE_SPACES.sub(' ', text)
    text = BAD_SYMBOLS.sub('', text)  # reemplaza lo que se indica anteriormente como mal simbolo
    text = text.replace('x', '')

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # elimina stopwords
    return text


# usa paquete cleantext
def cleaner(text: str) -> str:
    return clean(text,  # pylint: disable=too-many-arguments, too-many-branches
                 clean_all=False,
                 extra_spaces=True,
                 stemming=False,
                 stopwords=True,
                 lowercase=True,
                 numbers=True,
                 punct=True,
                 reg='',
                 reg_replace='',
                 stp_lang='english'  # set to 'de' for German special handling
                 )


def cleaner_stem(text: str) -> str:
    return clean(text,  # pylint: disable=too-many-arguments, too-many-branches
                 clean_all=True,
                 extra_spaces=False,
                 stemming=False,
                 stopwords=False,
                 lowercase=False,
                 numbers=False,
                 punct=False,
                 reg='',
                 reg_replace='',
                 stp_lang='english'  # set to 'de' for German special handling
                 )


def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([str(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize(text)])
