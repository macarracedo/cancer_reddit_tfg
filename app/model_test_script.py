# DataBase access
# Natural Language Processing
import nltk
# Data Manipulation
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# Statistical libraries
from sklearn.feature_selection import chi2
# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# Performance Evaluation and Support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sqlalchemy import select

import db
from models import Submission
# Clean text methods
from preprocessing import cleaner

# Data Visualisation
# import seaborn as sns # not installed yet

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

np.random.seed(500)

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
print(category_labels)

print("=======" * 15)

# De manera inversa, creo un diccionario para convertir etiquetas en categorias
category_reverse = dict(flair_category[['id', 'flair']].values)
print(category_reverse)

data['combined'] = data['title']  # Creo una columna combinando titulo y cuerpo, para aportar más información
count = 0
for i in range(len(data)):
    if type(data.loc[i]['body']) != float:
        data['combined'][i] = data['combined'][i] + ' ' + data['body'][i]

data.head(20)

# llama a una de las funciones definidas arriba
data['combined'] = data['combined'].apply(cleaner)

# creo instancia del vectorizador TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5,
                        norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 2))

# Extracting the features by fitting the Vectorizer on Combined Data
feat = tfidf.fit_transform(data['combined']).toarray()
labels = data['id']  # Series containing all the post labels
print(feat.shape)

# chisq2 statistical test
N = 5  # Number of examples to be listed
for f, i in sorted(category_labels.items()):
    chi2_feat = chi2(feat, labels == i)
    indices = np.argsort(chi2_feat[0])
    feat_names = np.array(tfidf.get_feature_names_out())[indices]
    unigrams = [w for w in feat_names if len(w.split(' ')) == 1]
    bigrams = [w for w in feat_names if len(w.split(' ')) == 2]
    print("\nFlair '{}':".format(f))
    print("Most correlated unigrams:\n\t. {}".format('\n\t. '.join(unigrams[-N:])))
    print("Most correlated bigrams:\n\t. {}".format('\n\t. '.join(bigrams[-N:])))

### Rescatar datos aquí del pickle

# Splitting 20% of the data into train test split

X_train, X_test, y_train, y_test = train_test_split(data['combined'], data['flair'],
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Creating an instance of the TFID transformer
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(X_train)

# Creating an instance of the TFID transformer
tfidf_trans = TfidfTransformer()
X_train_tfidf = tfidf_trans.fit_transform(X_train_counts)


# Naive Bayes Classifier
def nb_classifier(X_train, X_test, y_train, y_test):
    nb_fit = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('model', MultinomialNB()),
                       ])
    nb_fit.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = nb_fit.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
    precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
    recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
    recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')

    print(f"\n\nModel Accuracy: {acc}, Model F_Score_micro: {f_score_micro},"
          f"\nModel F_Score_macro: {f_score_macro},\nPrecision_micro: {precision_micro},"
          f"\nModel Precision_macro: {precision_macro},\nModel Recall_micro: {recall_micro}"
          f"\nModel Recall_macro: {recall_macro}")


# Random Forest Classifier
def random_forest(X_train, X_test, y_train, y_test):
    forest = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('model', RandomForestClassifier()),
                       ])
    forest.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = forest.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
    precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
    recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
    recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')

    print(f"\n\nModel Accuracy: {acc}, Model F_Score_micro: {f_score_micro},"
          f"\nModel F_Score_macro: {f_score_macro},\nPrecision_micro: {precision_micro},"
          f"\nModel Precision_macro: {precision_macro},\nModel Recall_micro: {recall_micro}"
          f"\nModel Recall_macro: {recall_macro}")


# Support Vector Machines Classifier
def svc(X_train, X_test, y_train, y_test):
    svc_fit = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', SVC()),
                        ])
    svc_fit.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = svc_fit.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
    precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
    recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
    recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')

    print(f"\n\nModel Accuracy: {acc}, Model F_Score_micro: {f_score_micro},"
          f"\nModel F_Score_macro: {f_score_macro},\nPrecision_micro: {precision_micro},"
          f"\nModel Precision_macro: {precision_macro},\nModel Recall_micro: {recall_micro}"
          f"\nModel Recall_macro: {recall_macro}")


# Logistic Regression Classifier
def log_reg(X_train, X_test, y_train, y_test):
    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('model', LogisticRegression()),
                       ])
    logreg.fit(X_train, y_train)  # Fitting the data to the trianing data

    # Making Predictions on the test data
    y_pred = logreg.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    f_score_micro = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    f_score_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    precision_micro = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
    precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
    recall_micro = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
    recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')

    print(f"\n\nModel Accuracy: {acc}, Model F_Score_micro: {f_score_micro},"
          f"\nModel F_Score_macro: {f_score_macro},\nPrecision_micro: {precision_micro},"
          f"\nModel Precision_macro: {precision_macro},\nModel Recall_micro: {recall_micro}"
          f"\nModel Recall_macro: {recall_macro}")


print("Evaluate Naive Bayes Classifier")
nb_classifier(X_train, X_test, y_train, y_test)

print("“Evaluate Random Forest Classifier”")
random_forest(X_train, X_test, y_train, y_test)

print("“Evaluate Logistic Regression Model”")
log_reg(X_train, X_test, y_train, y_test)

print("“Evaluate SVC Model”")
svc(X_train, X_test, y_train, y_test)
