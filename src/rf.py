import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib.legend_handler import HandlerLine2D

from src.data_loading import preprocess


class RF:

    def __init__(self):

        self.model = RandomForestClassifier()
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

        # params = {
        #     'bootstrap': [True],
        #     'class_weight': [None],
        #     'criterion': ['gini'],
        #     'max_depth': [300],
        #     'max_features': ['auto'],
        #     'max_leaf_nodes': [None],
        #     'min_samples_leaf': [20],
        #     'min_samples_split': [10],
        #     'n_estimators': [200, 400, 800],
        #     'oob_score': [True],
        #     'n_jobs': [-1],
        #     'random_state': [42]
        # }

        params = {
            'bootstrap': [True],
            'criterion': ['gini', 'entropy'],
            'max_depth': [250],
            'max_features': ['sqrt'],
            'min_samples_leaf': [0.2],
            'min_samples_split': [0.2],
            'n_estimators': [16, 32, 64, 100, 200],
            'n_jobs': [-1],
            'random_state': [42]
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

    def n_estimators_tuning(self, X_tr, y_tr, X_ts, y_ts):

        X_tr = self.count_vect.fit_transform(X_tr)
        X_tr = self.tfidf.fit_transform(X_tr)
        X_ts = self.count_vect.transform(X_ts)
        X_ts = self.tfidf.transform(X_ts)

        n_estimators = [32, 64, 100, 200, 400, 800, 1000, 2000]
        train_results = []
        test_results = []

        for estimator in n_estimators:
            self.model = RandomForestClassifier(min_samples_split=estimator)
            self.model.fit(X_tr, y_tr)

            train_pred = self.model.predict(X_tr)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_tr, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)

            y_pred = self.model.predict(X_ts)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_ts, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
        line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('N_estimators')
        plt.show()

    def depth_tuning(self, X_tr, y_tr, X_ts, y_ts):

        X_tr = self.count_vect.fit_transform(X_tr)
        X_tr = self.tfidf.fit_transform(X_tr)
        X_ts = self.count_vect.transform(X_ts)
        X_ts = self.tfidf.transform(X_ts)

        max_depths = np.linspace(1, 300, 32, endpoint=True)
        train_results = []
        test_results = []

        for max_depth in max_depths:
            self.model = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
            self.model.fit(X_tr, y_tr)

            train_pred = self.model.predict(X_tr)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_tr, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)

            y_pred = self.model.predict(X_ts)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_ts, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
        line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('Tree depth')
        plt.show()

    def min_samples_split_tuning(self, X_tr, y_tr, X_ts, y_ts):

        X_tr = self.count_vect.fit_transform(X_tr)
        X_tr = self.tfidf.fit_transform(X_tr)
        X_ts = self.count_vect.transform(X_ts)
        X_ts = self.tfidf.transform(X_ts)

        min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
        train_results = []
        test_results = []

        for min_samples_split in min_samples_splits:
            self.model = RandomForestClassifier(min_samples_split=min_samples_split)
            self.model.fit(X_tr, y_tr)

            train_pred = self.model.predict(X_tr)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_tr, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)

            y_pred = self.model.predict(X_ts)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_ts, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
        line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('Min samples split')
        plt.show()

    def min_samples_leafs_tuning(self, X_tr, y_tr, X_ts, y_ts):

        X_tr = self.count_vect.fit_transform(X_tr)
        X_tr = self.tfidf.fit_transform(X_tr)
        X_ts = self.count_vect.transform(X_ts)
        X_ts = self.tfidf.transform(X_ts)

        min_samples_leafs = np.linspace(0.1, 1.0, 5, endpoint=True)
        train_results = []
        test_results = []

        for min_samples_leaf in min_samples_leafs:
            self.model = RandomForestClassifier(min_samples_split=min_samples_leaf)
            self.model.fit(X_tr, y_tr)

            train_pred = self.model.predict(X_tr)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_tr, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)

            y_pred = self.model.predict(X_ts)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_ts, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
        line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('Min samples leaf')
        plt.show()

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

    rf_class = RF()

    # rf_class.n_estimators_tuning(X_train, y_train, X_test, y_test)
    # rf_class.depth_tuning(X_train, y_train, X_test, y_test)
    # rf_class.min_samples_split_tuning(X_train, y_train, X_test, y_test)
    # rf_class.min_samples_leafs_tuning(X_train, y_train, X_test, y_test)

    rf_class.train(X_train.values, y_train.values)
    rf_class.test(X_test.values, y_test.values)
    rf_class.save('RF_sentiment')

    # rf_class = RF.load('RF_sentiment')
    # rf_class.predict("Bring dildos asap, you fucking bitch, gonna ramp you up :P")

    df = pd.read_csv('../data/negative2.csv')
    df.dropna(subset=['comment_text'], inplace=True)
    print(df.isnull().sum())

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(df['comment_text'],
                                                                        df['toxicity_level'],
                                                                        random_state=42)

    rf_class_neg = RF()

    rf_class_neg.train(X_train_neg.values, y_train_neg.values)
    rf_class_neg.test(X_test_neg.values, y_test_neg.values)
    rf_class_neg.save('RF_toxicity_level')
