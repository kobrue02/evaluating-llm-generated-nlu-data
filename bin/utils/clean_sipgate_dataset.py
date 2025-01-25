import pandas as pd


def clean_sipgate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the sipgate dataset by removing duplicates, intents with too few samples and samples with low occurrences.
    """

    # deduplicate text column
    df = df.drop_duplicates(subset="text")

    # keep only intents with more than 20 and less than 1000 samples
    df = df.groupby("intent").filter(lambda x: len(x) > 100 and len(x) < 500)

    # keep only intents that don't start with "_"
    df = df[~df.intent.str.startswith("_")]

    # for each intent, keep the 100 samples with highest value in occurrences column
    df = df.groupby("intent").apply(lambda x: x.nlargest(100, "occurrences"))

    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    from bin.utils.read_datasets import read_sipgate_dataset
    df = read_sipgate_dataset()
    df = clean_sipgate_dataset(df)
    print(df.head())