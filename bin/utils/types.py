"""
Utility functions and classes for working with datasets and models.

Author: Konrad Brüggemann

University of Potsdam, 2024-2025

Part of the Bachelor's Thesis: "Evaluating and Improving the Synthetic Data Generation Abilities of LLMs for Realistic NLU Training Data"
"""

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import pickle
import pandas as pd


class DataSet:
    """
    A class to represent a dataset.
    Attributes:
        data (list): The data of the dataset.
        labels (list): The labels of the dataset.
    Methods:
        __init__(data, labels): Constructs a dataset with the given data and labels.
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data and label at the given index.
        __iter__(): Returns an iterator for the dataset.
    """

    def __init__(self, data=None, labels=None):
        self.data = data if data is not None else []
        self.labels = labels if labels is not None else []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"DataSet(data={self.data}, labels={self.labels})"

    def __str__(self):
        return f"DataSet(data={self.data}, labels={self.labels})"

    def __add__(self, other):
        return DataSet(self.data + other.data, self.labels + other.labels)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self.data += other.data
        self.labels += other.labels
        return self

    def __mul__(self, n):
        return DataSet(self.data * n, self.labels * n)

    def __rmul__(self, n):
        return self.__mul__(n)

    def __imul__(self, n):
        self.data *= n
        self.labels *= n
        return self

    def __eq__(self, other):
        return self.data == other.data and self.labels == other.labels

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, item):
        return item in self.data or item in self.labels

    def __reversed__(self):
        return DataSet(reversed(self.data), reversed(self.labels))

    def __copy__(self):
        return DataSet(self.data.copy(), self.labels.copy())

    def __deepcopy__(self, memo):
        return DataSet(self.data.copy(), self.labels.copy())

    def append(self, data, label):
        self.data.append(data)
        self.labels.append(label)

    def extend(self, data, labels):
        self.data.extend(data)
        self.labels.extend(labels)

    def insert(self, idx, data, label):
        self.data.insert(idx, data)
        self.labels.insert(idx, label)

    def remove(self, data, label):
        self.data.remove(data)
        self.labels.remove(label)

    def pop(self, idx=-1):
        return self.data.pop(idx), self.labels.pop(idx)

    def clear(self):
        self.data.clear()
        self.labels.clear()

    def count(self, item):
        return self.data.count(item) + self.labels.count(item)

    def index(self, item, start=0, stop=None):
        if stop is None:
            stop = len(self)
        try:
            return self.data.index(item, start, stop)
        except ValueError:
            return self.labels.index(item, start, stop)

    def reverse(self):
        self.data.reverse()
        self.labels.reverse()

    def sort(self, key=None, reverse=False):
        if key is None:

            def key(x):
                return x

        self.data.sort(key=key, reverse=reverse)
        self.labels.sort(key=key, reverse=reverse)

    def copy(self):
        return DataSet(self.data.copy(), self.labels.copy())

    def deepcopy(self):
        return DataSet(self.data.copy(), self.labels.copy())

    def shuffle(self):
        data, labels = shuffle(self.data, self.labels)
        return DataSet(data, labels)

    def split(self, train_size=0.8, test_size=None, random_state=None):
        if test_size is None:
            test_size = 1 - train_size
        data_train, data_test, labels_train, labels_test = train_test_split(
            self.data,
            self.labels,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        return DataSet(data_train, labels_train), DataSet(data_test, labels_test)

    def to_tensors(self, tokenizer, max_length=512, padding=True, truncation=True):
        data = tokenizer(
            self.data,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        labels = torch.tensor(self.labels)
        return data, labels

    def to_data_frame(self, columns: list[str] = None):
        if columns is None:
            columns = ["data", "labels"]
        return pd.DataFrame({columns[0]: self.data, columns[1]: self.labels})

    def to_csv(self, path, columns=None, **kwargs):
        self.to_data_frame(columns).to_csv(path, **kwargs)

    def to_json(self, path, columns=None, **kwargs):
        self.to_data_frame(columns).to_json(path, **kwargs)

    def to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_tensor_dataset(cls, dataset, tokenizer):
        return cls.from_tensors(*dataset.tensors, tokenizer)

    @classmethod
    def from_tensors(cls, data, labels, tokenizer):
        data = tokenizer.convert_ids_to_tokens(data["input_ids"])
        labels = labels.numpy()
        return cls(data, labels)

    @classmethod
    def from_data_frame(cls, df, columns=None):
        if columns is None:
            columns = ["data", "labels"]
        return cls(df[columns[0]].tolist(), df[columns[1]].tolist())

    @classmethod
    def from_csv(cls, path, columns=None, **kwargs):
        return cls.from_data_frame(pd.read_csv(path, **kwargs), columns)

    @classmethod
    def from_json(cls, path, columns=None, **kwargs):
        return cls.from_data_frame(pd.read_json(path, **kwargs), columns)

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_text_file(cls, path):
        with open(path, "r") as f:
            data = f.read().splitlines()
        return cls(data, [None] * len(data))

    @classmethod
    def from_text_files(cls, *paths):
        data, labels = [], []
        for path in paths:
            with open(path, "r") as f:
                data.extend(f.read().splitlines())
            labels.extend([None] * len(data))
        return cls(data, labels)

    @classmethod
    def from_text(cls, text):
        return cls(text.splitlines(), [None] * len(text.splitlines()))

    @classmethod
    def from_texts(cls, *texts):
        data, labels = [], []
        for text in texts:
            data.extend(text.splitlines())
            labels.extend([None] * len(data))
        return cls(data, labels)

    @classmethod
    def from_dict(cls, d):
        return cls(d["data"], d["labels"])

    def to_dict(self):
        return {"data": self.data, "labels": self.labels}
