import pandas as pd


def read_sipgate_dataset():
    full_df = pd.read_csv("data/sipgate_data.csv")
    dataset = full_df[["text", "phraseIntent"]].copy()
    dataset.rename(columns={"phraseIntent": "intent"}, inplace=True)
    return dataset


if __name__ == "__main__":
    df = read_sipgate_dataset()
    print(df.head())
