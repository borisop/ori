import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords

TARGET_COLS = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
X_COLS = ['comment_text']

TOXICITY_LEVEL = {
    'severe_toxicity': 1,
    'obscene': 2,
    'identity_attack': 3,
    'insult': 4,
    'threat': 5,
    'positive': 0
}

X_COUNT = 8000
CHUNK_SIZE = 20000
N_ROWS = 300000

def label_preprocessing_all(df):

    toxicity_level = []
    sentiment = []

    for index, row in df.iterrows():

        if row['severe_toxicity'] > row['obscene'] \
                and row['severe_toxicity'] > row['identity_attack'] \
                and row['severe_toxicity'] > row['insult'] \
                and row['severe_toxicity'] > row['threat']:
            toxicity_level.append(TOXICITY_LEVEL['severe_toxicity'])
            sentiment.append(1)

        elif row['obscene'] > row['severe_toxicity'] \
                and row['obscene'] > row['identity_attack'] \
                and row['obscene'] > row['insult'] \
                and row['obscene'] > row['threat']:
            toxicity_level.append(TOXICITY_LEVEL['obscene'])
            sentiment.append(1)

        elif row['identity_attack'] > row['severe_toxicity'] \
                and row['identity_attack'] > row['obscene'] \
                and row['identity_attack'] > row['insult'] \
                and row['identity_attack'] > row['threat']:
            toxicity_level.append(TOXICITY_LEVEL['identity_attack'])
            sentiment.append(1)

        elif row['insult'] > row['severe_toxicity'] \
                and row['insult'] > row['obscene'] \
                and row['insult'] > row['identity_attack'] \
                and row['insult'] > row['threat']:
            toxicity_level.append(TOXICITY_LEVEL['insult'])
            sentiment.append(1)

        elif row['threat'] > row['severe_toxicity'] \
                and row['threat'] > row['obscene'] \
                and row['threat'] > row['identity_attack'] \
                and row['threat'] > row['insult']:
            toxicity_level.append(TOXICITY_LEVEL['threat'])
            sentiment.append(1)
        else:
            toxicity_level.append(0)
            sentiment.append(0)

    df['toxicity_level'] = toxicity_level
    df['sentiment'] = sentiment
    df.drop(columns=TARGET_COLS, inplace=True)

    return df


def quantitize_data(chunks):

    def asd(chunk, key):
        q = min(len(chunk), quantities[key]['q'])
        quantities[key]['df'].append(chunk[:q])
        quantities[key]['q'] -= q

    quantities = {
        'severe_toxicity': {'q': X_COUNT, 'df': []},
        'obscene': {'q': X_COUNT, 'df': []},
        'identity_attack': {'q': X_COUNT, 'df': []},
        'insult': {'q': X_COUNT, 'df': []},
        'threat': {'q': X_COUNT, 'df': []},
        'positive': {'q': 0, 'df': []},
    }

    for c in chunks:

        if quantities['severe_toxicity']['q'] > 0:
            asd(c[c.toxicity_level == 1], 'severe_toxicity')

        if quantities['obscene']['q'] > 0:
            asd(c[c.toxicity_level == 2], 'obscene')

        if quantities['identity_attack']['q'] > 0:
            asd(c[c.toxicity_level == 3], 'identity_attack')

        if quantities['insult']['q'] > 0:
            asd(c[c.toxicity_level == 4], 'insult')

        if quantities['threat']['q'] > 0:
            asd(c[c.toxicity_level == 5], 'threat')

        if quantities['positive']['q'] > 0:
            asd(c[c.toxicity_level == 0], 'positive')

    dfs = []
    for k, v in quantities.items():
        dfs.extend(v['df'])

    ret_df = pd.concat(dfs)
    ret_df.reset_index(inplace=True, drop=True)
    return ret_df


def load(path):
    chunks = pd.read_csv(path, usecols=X_COLS+TARGET_COLS, chunksize=CHUNK_SIZE, nrows=N_ROWS, skiprows=[i for i in range(1, 200000)])
    chunks = list(map(lambda chunk: label_preprocessing_all(chunk), chunks))
    ret_df = quantitize_data(chunks)
    return ret_df


def preprocess(text):

    # nltk.download('punkt')
    # nltk.download('wordnet')

    text = text.strip().lower()
    text = re.sub(r'\d+', '', text)

    tokens = nltk.word_tokenize(text)
    text = " ".join([t for t in tokens if t not in set(stopwords.words('english'))])
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    text = " ".join(lemmatizer.lemmatize(token, 'v') for token in tokens)

    return text


if __name__ == '__main__':
    df = load('../data/train.csv')
    df.to_csv('../data/negative1.csv', index=False)
    # df = pd.read_csv('../old/p1.csv')
    df['comment_text'] = df['comment_text'].map(lambda row: preprocess(row))

    df.dropna(subset=['comment_text'], inplace=True)
    print(df.isnull().sum())

    for name, code in TOXICITY_LEVEL.items():
        count = len(df[df['toxicity_level'] == code])
        print('{}:\t\t{}'.format(name, count))

    df.to_csv('../data/negative2.csv', index=False)


