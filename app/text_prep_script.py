# DataBase access

# Parser cmd-line options and arguments
import argparse
# File manage
import pickle
from pathlib import Path

# Data Manipulation
import numpy as np
import pandas as pd
# Natural Language Processing
from sklearn.feature_extraction.text import TfidfVectorizer
# Statistical libraries
from sklearn.feature_selection import chi2
from sqlalchemy import select

# DataBase access
import db
from models import Submission
# Clean text methods
from preprocessing import cleaner
from preprocessing import cleaner_stem
from preprocessing import lemmatize_text

# Data Manipulation
# Statistical libraries
# Natural Language Processing
# Clean text methods
# Data Visualisation
# import seaborn as sns # not installed yet

# Machine Learning

# Performance Evaluation and Support

p = Path()
p = f'{p.home()}/Documentos/cancer_reddit_tfg'
filename = f"{str(p)}/data/prep_tf-idf.p"

parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                             '\n preprocess its body and'
                                             '\n saves them to a pickle file.')

group_args = parser.add_mutually_exclusive_group()
group_args.add_argument('-l', '--lemmatization', action='store_true', help='Reduce las palablas a su lema')
group_args.add_argument('-s', '--stemming', action='store_true', help='Reduce las palabras a su raiz')

args = parser.parse_args()
print(args)

db.Base.metadata.create_all(db.engine)
query = select(Submission.link_flair_text, Submission.title, Submission.selftext).where(
    Submission.link_flair_text != None)
result = db.session.execute(query).fetchall()

features = [
    'flair',
    'title',
    'body'
]
data = pd.DataFrame(result, columns=features)

data['id'] = data['flair'].factorize()[0]
flair_category = data[['flair', 'id']].drop_duplicates().sort_values('id')
print(flair_category)

# Creo un diccionario de etiquetas
category_labels = dict(flair_category.values)
print(f'Category_lables: {category_labels}')

# Combino titulo y cuerpo en una única columna para aportar más información
data['combined'] = data['title']
for i in range(len(data)):
    if type(data.loc[i]['body']) != float:
        data['combined'][i] = data['combined'][i] + ' ' + data['body'][i]

data.head(20)

# manage stopwords, lowercase, extra spaces, punct, numbers... with cleantext
data['result'] = data['combined'].apply(cleaner)
print(f"Combined Column: \n{data['combined']}")

print(f"\n\nAfter cleaner (stopwords, lowercase, extra spaces, punct, numbers...): \n{data['result']}")

# filtrado de stopwords con nltk no aporta nada al hablerlo hecho antes con cleantext

if args.lemmatization:
    data['result'] = data['result'].apply(lemmatize_text)  # no noto que haga nada
    filename = f"{str(p)}/data/prep_lemm_tf-idf.p"
elif args.stemming:
    data['result'] = data['result'].apply(cleaner_stem)
    filename = f"{str(p)}/data/prep_stem_tf-idf.p"

print(f"\n\nAfter lemmatization/stemming (if asked for): \n{data['result']}")

# creo instancia del vectorizador TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5,
                        norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 2))

# Extraigo los n-gramas ejecutando el vectorizador tf-idf sobre la columna 'result'
feat = tfidf.fit_transform(data['result'])
labels = data['id']  # Series containing all the post labels

# chisq2 statistical test
to_save = dict()
for f, i in category_labels.items():
    chi2_feat = chi2(feat, labels == i)
    indices = np.argsort(chi2_feat[0])
    feat_names = np.array(tfidf.get_feature_names_out())[indices]
    to_save[f] = feat_names

pickle.dump(to_save, open(filename, "wb"))
