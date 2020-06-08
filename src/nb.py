import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from src.data_loading import preprocess


class NB:

    def __init__(self):

        self.model = MultinomialNB()
        self.tfidf = TfidfTransformer()
        self.count_vect = CountVectorizer()
        self.best_params = {}
        self.train_report = ""
        self.train_f1 = []
        self.train_acc = []
        self.test_report = ""
        self.test_f1 = []
        self.test_acc = []

    def train(self, X, y):
        X = self.count_vect.fit_transform(X)
        X = self.tfidf.fit_transform(X)

        params = {
            'alpha': [.01, .05, .1, .2, .5, 1]
        }

        grid_search = GridSearchCV(self.model,
                                   param_grid=params,
                                   scoring='f1_micro',
                                   n_jobs=-1,
                                   refit=True,
                                   cv=3,
                                   verbose=2)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        y_pred = self.model.predict(X)
        report = classification_report(y_true=y, y_pred=y_pred)
        f1 = f1_score(y_true=y, y_pred=y_pred, average='micro')
        acc = balanced_accuracy_score(y_true=y, y_pred=y_pred)
        self.train_report = report
        self.train_f1 = f1
        self.train_acc = acc
        print(report)
        print('f1 score:  {}'.format(f1))
        print('acc score: {}'.format(acc))
        print('\n\t\tFINISHED TRAINING')

    def test(self, X, y):
        X = self.count_vect.transform(X)
        X = self.tfidf.transform(X)

        y_pred = self.model.predict(X)
        report = classification_report(y_true=y, y_pred=y_pred)
        f1 = f1_score(y_true=y, y_pred=y_pred, average='micro')
        acc = balanced_accuracy_score(y_true=y, y_pred=y_pred)
        self.test_report = report
        self.test_f1 = f1
        self.test_acc = acc
        print(report)
        print('f1 score:  {}'.format(f1))
        print('acc score: {}'.format(acc))
        print('\n\t\tFINISHED TESTING')

    def save(self, name):
        filename1 = '../models/{}.pkl'.format(name)
        pickle.dump(self, open(filename1, 'wb'))
        print('Saved to {}'.format(filename1))

    @classmethod
    def load(cls, name):
        filename1 = '../models/{}.pkl'.format(name)
        with open(filename1, 'rb') as file:
            return pickle.load(file)

    def predict(self, comment):
        preprocessed_comment = preprocess(comment)
        X = self.count_vect.transform([preprocessed_comment])
        X = self.tfidf.transform(X)
        cls = self.model.predict(X)
        print('\n    "{}"\n -> "{}"\nis predicted as {}\n'.format(comment, preprocessed_comment, cls))
        return cls

    def print_stats(self):
        print('Best params: {}'.format(self.best_params))
        print('\n\tTRAIN SCORES\n')
        print(self.train_report)
        print('f1 score:    {}'.format(self.train_f1))
        print('acc score:   {}'.format(self.train_acc))
        print('\n\tTEST SCORES\n')
        print(self.test_report)
        print('f1 score:    {}'.format(self.test_f1))
        print('acc score:   {}'.format(self.test_acc))


if __name__ == '__main__':
    df = pd.read_csv('../data/positive2.csv')
    df.dropna(subset=['comment_text'], inplace=True)
    print(df.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['sentiment'], random_state=42)

    nb_class = NB()

    nb_class.train(X_train.values, y_train.values)
    nb_class.test(X_test.values, y_test.values)
    nb_class.save('NB_sentiment')

    # nb_class = NB.load('NB_sentiment')
    # nb_class.predict("Bring dildos asap, you fucking bitch, gonna ramp you up :P")

    df = pd.read_csv('../data/negative2.csv')
    df.dropna(subset=['comment_text'], inplace=True)
    print(df.isnull().sum())

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(df['comment_text'], df['toxicity_level'], random_state=42)

    nb_class_neg = NB()

    nb_class_neg.train(X_train_neg.values, y_train_neg.values)
    nb_class_neg.test(X_test_neg.values, y_test_neg.values)
    nb_class_neg.save('NB_toxicity_level')


