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

# Text preprocessing


# Natural Language Processing

# DataBase access

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

raise Exception
# Recojo submissions de la base de datos a un dataframe
db.Base.metadata.create_all(db.engine)
query = sqlalchemy.select(Submission.link_flair_text, Submission.title, Submission.selftext).where(
    Submission.link_flair_text is not None)
result = db.session.execute(query).fetchall()

features = [
    'flair',
    'title',
    'body'
]
prep_df = pd.DataFrame(result, columns=features)

prep_df['combined'] = prep_df['title']  # Creo una columna combinando titulo y cuerpo, para aportar más información

for i in range(len(prep_df)):
    if type(prep_df.loc[i]['body']) != float:
        prep_df['combined'][i] = prep_df['combined'][i] + ' ' + prep_df['body'][i]
# Elimino columnas de Titulo y Cuerpo
prep_df.drop('title', inplace=True, axis=1)
prep_df.drop('body', inplace=True, axis=1)

# Aquí realizo el preprocesado parametrizado en funión de los argumentos...


#   1. Uso cleantext para remover símbolos de puntuación, caracteres no ASCII, URL, emails, dígitos, minusculas, etc.
prep_df['result'] = prep_df['combined'].apply(cln)
prep_df['flair'] = prep_df['flair'].apply(str.lower)

#   2. Uso NLTK para tokenización.
prep_df['result'].apply(word_tokenize)
#   3. Uso NLTK para lemmatización o stemmización. NO FUNCIONA NINGUNO
if args.lemmatization:
    lm = WordNetLemmatizer()
    prep_df['result'] = prep_df['result'].apply(lm.lemmatize)

elif args.stemming:
    ps = PorterStemmer()
    prep_df['result'] = prep_df['result'].apply(ps.stem)

# Visualización y guardado del resultado
print(f"Result DataFrame: \n{prep_df}")

# Finalmente guardo el dataframe preprocesado en un pickle.
prep_df.to_pickle(f'{data_path}/{filename}')
print(f'saved prep_df in {data_path}/{filename}')
