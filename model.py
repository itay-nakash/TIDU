
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel # install with pip install statsmodels


MODE = 'ordinal' # 'ordinal' or 'regression' or 'classification'
assert MODE in ['ordinal', 'regression', 'classification'], "MODE must be one of 'ordinal', 'regression', 'classification'"

data_folder= 'data/'


# adding multi class clasification, and additional analyzing according to this

class TextClassifier(ABC):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, sublinear_tf=True,
                                          max_features=40, stop_words='english')

    @abstractmethod
    def get_classifier(self):
        pass

    def load_data(self):
        train_data = pd.read_csv(data_folder + 'train.csv')
        val_data = pd.read_csv(data_folder + 'val.csv')
        full_train_data = pd.concat([train_data, val_data])

        self.X_train = full_train_data['text']
        self.y_train = full_train_data['label']
        test_data = pd.read_csv(data_folder + 'test.csv')
        self.X_test = test_data['text']
        self.y_test = test_data['label']

    def transform_data(self):
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)

    def train(self):
        self.clf = self.get_classifier()
        self.clf.fit(self.X_train_vec, self.y_train)

    def predict(self):
        return self.clf.predict(self.X_test_vec)

    def majority_vote_baseline(self):
        most_common_class = Counter(self.y_train).most_common(1)[0][0]
        return [most_common_class] * len(self.y_test)


class SVMTextClassifier(TextClassifier):

    def get_classifier(self):
        return LinearSVC()


class RandomForestTextClassifier(TextClassifier):

    def get_classifier(self):
        return RandomForestClassifier()


class NaiveBayesTextClassifier(TextClassifier):

    def get_classifier(self):
        return MultinomialNB()


class LinearRegressionTextClassifier(TextClassifier):

    def get_classifier(self):
        return LinearRegression()


def print_report(y_test, y_pred, method):
    print(f'{method} Results:')
    print(classification_report(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')
    plt.show()


def print_top_features(vectorizer, clf, class_labels, n=10):
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(class_labels):
        top_features = np.argsort(clf.coef_[i])[-n:]
        print(f"Top features for class {class_label}:")
        print(", ".join(feature_names[j] for j in top_features))


def plot_wordclouds(X_train, y_train, vectorizer, class_labels):
    for class_label in class_labels:
        class_words = ' '.join(X_train[y_train == class_label])
        class_word_vector = vectorizer.transform([class_words])
        words_freq = dict(zip(vectorizer.get_feature_names_out(), np.asarray(class_word_vector.sum(axis=0)).ravel()))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Most frequent words in class {class_label}")
        plt.show()


def ordinal_regression(X_train, y_train, X_test):
    # count number of constant columns  
    assert np.sum(np.all(X_train == X_train[0,:], axis = 0)) == 0, "X has constant columns"
    # Fit the ordered model
    model = OrderedModel(exog=X_train, endog=y_train,distr='logit', hasconst=False)
    result = model.fit()

    y_pred = result.predict(exog=X_test)
    # get argmax per row
    y_pred = np.argmax(y_pred, axis=1) + 1
    return y_pred

if __name__ == '__main__':
    
    if MODE == 'regression':
        classifier = LinearRegressionTextClassifier()

    if MODE == 'classification':
        classifier = SVMTextClassifier()

    if MODE == 'ordinal':
        classifier = SVMTextClassifier() # JUST FOR THE DATA PREPROCESSING
        
    classifier.load_data()
    classifier.transform_data()
  
    if MODE == 'ordinal':
        y_pred = ordinal_regression(X_train = classifier.X_train_vec.toarray(),
                                    y_train=classifier.y_train, 
                                    X_test = classifier.X_test_vec.toarray())
    
    
    
    classifier.train()
    y_pred = classifier.predict().round()
    baseline_pred = classifier.majority_vote_baseline()

    print_report(classifier.y_test, y_pred, 'TF-IDF & LinearSVC')
    print_report(classifier.y_test, baseline_pred, 'Majority Vote Baseline')

    plot_confusion_matrix(classifier.y_test, y_pred, labels=[1, 2, 3, 4])
    plot_confusion_matrix(classifier.y_test, baseline_pred, labels=[1, 2, 3, 4])

    print_top_features(classifier.vectorizer, classifier.clf, class_labels=[1, 2, 3, 4])

    plot_wordclouds(classifier.X_train, classifier.y_train, classifier.vectorizer, class_labels=[1, 2, 3, 4])


