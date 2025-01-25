import pandas as pd


def read_sipgate_dataset() -> pd.DataFrame:
    """
    Read the sipgate dataset and return a DataFrame with the specified columns.
    If columns is None, return the full DataFrame.
    """
    dataset = pd.read_csv("data/sipgate_data.csv")
    dataset.rename(columns={"phraseIntent": "intent"}, inplace=True)
    dataset = dataset[
        ["text", "intent", "occurrences", "phraseEntTypes", "annotations"]
    ]
    return dataset


if __name__ == "__main__":
    df = read_sipgate_dataset()
    print(df.head())
