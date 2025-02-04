import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from collections import Counter


class IntentClassifier:
    def __init__(self, vectorizer=None, model=None):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        if vectorizer:
            self.vectorizer = vectorizer
        if model:
            self.model = model

    def split_dataset(self, df: pd.DataFrame) -> tuple:
        """
        Splits the dataset into train and test sets ensuring each intent has 5 samples in the test set.

        Args:
            df (pd.DataFrame): DataFrame with columns 'text' and 'intent'.

        Returns:
            train_df (pd.DataFrame): Training set.
            test_df (pd.DataFrame): Test set.
        """
        test_dfs = []
        train_dfs = []

        df = df.dropna()
        for intent, group in df.groupby("intent"):
            if len(group) >= 5:
                test_samples = group.sample(5, random_state=42)
                train_samples = group.drop(test_samples.index)
            else:
                test_samples = group
                train_samples = pd.DataFrame()

            test_dfs.append(test_samples)
            train_dfs.append(train_samples)

        test_df = pd.concat(test_dfs)
        train_df = pd.concat(train_dfs)

        return train_df, test_df

    def fit(self, train_df):
        """
        Fits the classifier on the training data.

        Args:
            train_df (pd.DataFrame): Training data with columns 'text' and 'intent'.
        """
        X_train = self.vectorizer.fit_transform(train_df["text"])
        y_train = train_df["intent"]
        self.model.fit(X_train, y_train)

    def predict(self, texts):
        """
        Predicts intents for given texts.

        Args:
            texts (list of str): List of input texts.

        Returns:
            list: Predicted intents.
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
    
    def predict_proba(self, test_df):
        X_test = self.vectorizer.transform(test_df["text"])
        y_test = test_df["intent"]
        y_pred = self.model.predict(X_test)
        return y_test, y_pred


    def classification_report(self, test_df):
        """
        Evaluates the model on the test data.

        Args:
            test_df (pd.DataFrame): Test data with columns 'text' and 'intent'.

        Returns:
            str: Classification report.
        """
        y_test, y_pred = self.predict_proba(test_df)
        return classification_report(y_test, y_pred)
    
    def evaluate(self, test_df):
        """
        Evaluates the model on the test data.

        Args:
            test_df (pd.DataFrame): Test data with columns 'text' and 'intent'.

        Returns:
            dict: Classification report.
        """
        y_test, y_pred = self.predict_proba(test_df)
        return classification_report(y_test, y_pred, output_dict=True)
    
    def reset(self):
        """
        Resets the model.
        """
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()


if __name__ == "__main__":
    df = pd.read_csv("output/zero_shot_simple_data.csv")
    model = IntentClassifier()
    train_df, test_df = model.split_dataset(df)
    model.fit(train_df)
    report = model.evaluate(test_df)
    print(report)
