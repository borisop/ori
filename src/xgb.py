import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from src.data_loading import preprocess


class XGB:

    def __init__(self):

        self.model = XGBClassifier()
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
            'n_estimators': [1500, 2000],
            'max_depth': [4, 6, 8],
            'min_child_weight': [6, 8, 10],
            'gamma': [0],
            'learning_rate': [0.1, 0.7],
            'subsample': [0.5, 0.8]
        }

        grid_search = GridSearchCV(self.model,
                                   param_grid=params,
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

    def test(self, X, y):

        X = self.count_vect.transform(X)
        X = self.tfidf.transform(X)

        y_pred = self.model.predict(X)
        report = classification_report(y_true=y, y_pred=y_pred)
        f1 = f1_score(y_true=y, y_pred=y_pred , average='micro')
        acc = balanced_accuracy_score(y_true=y, y_pred=y_pred)
        self.test_report = report
        self.test_f1 = f1
        self.test_acc = acc
        print(report)
        print('f1 score:  {}'.format(f1))
        print('acc score: {}'.format(acc))

    def save(self, name):

        filename = '../models/{}.pkl'.format(name)
        pickle.dump(self, open(filename, 'wb'))
        print('Saved to {}'.format(filename))

    @classmethod
    def load(cls, name):

        filename = '../models/{}.pkl'.format(name)
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def predict(self, comment):

        preprocessed_comment = preprocess(comment)
        X = self.count_vect.transform([comment])
        X = self.tfidf.transform(X)
        cls = self.model.predict(X)
        print('\n"{}" -> "{}"\nis predicted as {}\n'.format(comment, preprocessed_comment, cls))
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

    xgb_class = XGB()

    xgb_class.train(X_train, y_train)
    xgb_class.test(X_test.values, y_test.values)
    xgb_class.save('XGB_sentiment_new')

    df = pd.read_csv('../data/negative2.csv')
    df.dropna(subset=['comment_text'], inplace=True)
    print(df.isnull().sum())

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(df['comment_text'], df['toxicity_level'], random_state=42)

    xgb_class_neg = XGB()

    xgb_class_neg.train(X_train_neg, y_train_neg)
    xgb_class_neg.test(X_test_neg.values, y_test_neg.values)
    xgb_class_neg.save('XGB_toxicity_level_new')

