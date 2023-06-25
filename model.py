import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

data_folder= '/home/itay.nakash/hw/roy_corse/data/'

class TextClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.clf = LinearSVC()

    def load_data(self):
        train_data = pd.read_csv(data_folder + 'train.csv')
        val_data = pd.read_csv(data_folder + 'val.csv')
        full_train_data = pd.concat([train_data, val_data])

        self.X_train = full_train_data['text']
        self.y_train = full_train_data['label']
        test_data= pd.read_csv(data_folder+'test.csv')
        self.X_test = test_data['text']
        self.y_test = test_data['label']

    def transform_data(self):
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)

    def train(self):
        self.clf.fit(self.X_train_vec, self.y_train)

    def predict(self):
        return self.clf.predict(self.X_test_vec)

    def majority_vote_baseline(self):
        most_common_class = Counter(self.y_train).most_common(1)[0][0]
        return [most_common_class] * len(self.y_test)


def print_report(y_test, y_pred, method):
    print(f'{method} Results:')
    print(classification_report(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def print_top_features(vectorizer, clf, class_labels, n=10):
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(class_labels):
        top_features = np.argsort(clf.coef_[i])[-n:]
        print(f"Top features for class {class_label}:")
        print(", ".join(feature_names[j] for j in top_features))


if __name__ == '__main__':
    classifier = TextClassifier()

    # Load and transform data
    classifier.load_data()
    classifier.transform_data()

    # Train classifier and get predictions
    classifier.train()
    y_pred = classifier.predict()

    # Get baseline predictions
    baseline_pred = classifier.majority_vote_baseline()

    # Print reports
    print_report(classifier.y_test, y_pred, 'TF-IDF & LinearSVC')
    print_report(classifier.y_test, baseline_pred, 'Majority Vote Baseline')

    # Plot confusion matrices
    plot_confusion_matrix(classifier.y_test, y_pred, labels=[1, 2, 3, 4])
    plot_confusion_matrix(classifier.y_test, baseline_pred, labels=[1, 2, 3, 4])

    # Print top features for each class
    print_top_features(classifier.vectorizer, classifier.clf, class_labels=[1, 2, 3, 4])
