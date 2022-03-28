"""
 text_prep_script.py
 Este script coge submissions de la BD, lleva a cabo un preprocesado del texto
 parametrizable. Esto es, se podrán modificar mediante argumentos los "filtros" que se le aplican al texto
 (stopwords, numeros, puntuación, apóstrofes, minúsculas, lematización o stemización, etc.).
 Seguidamente guardará el dataframe preprocesado en un pickle.
"""

# Parser cmd-line options and arguments
import argparse
# File manage
from pathlib import Path

# Data Manipulation
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

p = Path.cwd()
print(str(p))

filename = f"{str(p)}/data/prep_tf-idf.p"
data_path = f"{str(p)}/data"

parser = argparse.ArgumentParser(description='Takes submissions from database,'
                                             '\n preprocess its body and'
                                             '\n saves them to a pickle file.')
parser.add_argument('--output-filename', type=str, help='Name of the output file')
parser.add_argument('--extra_spaces', nargs='?', metavar='', const=False, type=bool, default=True,
                    help='Cleantext removes extra spaces')
parser.add_argument('--lowercase', nargs='?', const=False, type=bool, default=True, help='Cleantext sets lowercase')
parser.add_argument('--numbers', nargs='?', const=False, type=bool, default=True, help='Cleantext removes numbers')
parser.add_argument('--punct', nargs='?', const=False, type=bool, default=True, help='Cleantext removes punct')
parser.add_argument('--stopwords', nargs='?', const=False, type=bool, default=True, help='Cleantext removes stopwords')

group_args = parser.add_mutually_exclusive_group()
group_args.add_argument('-l', '--lemmatization', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palablas a su lema')
group_args.add_argument('-s', '--stemming', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palabras a su raiz')

args = parser.parse_args()
print(args)

filename = "prep_text.p" if args.output_filename is None else args.output_filename

print(f'Input file prep_df in {data_path}/{filename}')
tf_idf_df = pd.read_pickle(f'{data_path}/{filename}')

print(tf_idf_df.head(20))

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(tf_idf_df['combined'])

# get the first vector out (for the first document)
first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(),
                  index=tfidf_vectorizer.get_feature_names(),
                  columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False)

print(df)
print(df.T)

print(15 * '=====')

tf_idf_vect = TfidfVectorizer()
X_train_tf_idf = tf_idf_vect.fit_transform(tf_idf_df['combined']).toarray()
terms = tf_idf_vect.get_feature_names()

print(terms)
print(X_train_tf_idf)
