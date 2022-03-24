import pickle
from pathlib import Path

if __name__ == '__main__':
    p = Path()
    p = f'{p.home()}/Documentos/cancer_reddit_tfg'

    file_to_read = open(f"{str(p)}/data/prep_tf-idf.p", "rb")

    loaded_dictionary = pickle.load(file_to_read)

    N = 5  # Number of anagrams to display

    for k, values in loaded_dictionary.items():
        print(f"\nFlair '{k}'")
        unigrams = [w for w in values if len(w.split(' ')) == 1]
        bigrams = [w for w in values if len(w.split(' ')) == 2]
        print("Most correlated unigrams:\n\t. {}".format('\n\t. '.join(unigrams[-N:])))
        print("Most correlated bigrams:\n\t. {}".format('\n\t. '.join(bigrams[-N:])))
