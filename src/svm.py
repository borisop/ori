import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from src.data_loading import preprocess


class SVM:

    def __init__(self):

        self.model = SVC()
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
            'kernel': ['linear', 'poly'],
            'gamma': [0.1, 1, 10],
            'C': [0.1, 1, 10, 100],
            'degree': [3, 4, 6]
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
        X = self.count_vect.transform([preprocessed_comment])
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

    svm_class = SVM()

    svm_class.train(X_train, y_train)
    svm_class.test(X_test.values, y_test.values)
    svm_class.save('SVM_sentiment')

    df = pd.read_csv('../data/negative2.csv')
    df.dropna(subset=['comment_text'], inplace=True)
    print(df.isnull().sum())

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(df['comment_text'], df['toxicity_level'], random_state=42)

    svm_class_neg = SVM()

    svm_class_neg.train(X_train_neg, y_train_neg)
    svm_class_neg.test(X_test_neg.values, y_test_neg.values)
    svm_class_neg.save('SVM_toxicity_level')