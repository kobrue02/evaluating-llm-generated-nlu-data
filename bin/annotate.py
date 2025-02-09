from bin.utils.methods import clean_synthetic_dataset, load_df
import pandas as pd
import csv
import os


def annotate(input_file: str, output_file: str):
    df = load_df(input_file)
    df = clean_synthetic_dataset(df)
    # select one sample from each intent
    df = df.groupby("intent").first().reset_index()
    
    # Create new columns for annotations
    df["Linguistic Naturalness"] = None
    df["Semantic Appropriateness"] = None
    df["Overall Quality"] = None

    # Iterate over the 'text' column for annotation
    for index, row in df.iterrows():
        print("\n", row["intent"])
        print("Sample:", row["text"])
        
        # Get user input for each category
        try:
            ln = int(input("Linguistic Naturalness (1-5): "))
            sa = int(input("Semantic Appropriateness (1-5): "))
            oq = int(input("Overall Quality (1-5): "))

            # Validate input
            if not (1 <= ln <= 5 and 1 <= sa <= 5 and 1 <= oq <= 5):
                print("Invalid input! Ratings must be between 1 and 5. Skipping this sample.")
                continue

            # Store annotations in DataFrame
            df.at[index, "Linguistic Naturalness"] = ln
            df.at[index, "Semantic Appropriateness"] = sa
            df.at[index, "Overall Quality"] = oq

        except ValueError:
            print("Invalid input! Please enter integers between 1 and 5. Skipping this sample.")

        # Save progress after each annotation
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"\nAnnotation completed. Results saved to {output_file}.")


if __name__ == "__main__":

    dirs = ["llama", "phi"]
    for d in dirs:
        # get all csv files in the directory
        files = [f for f in os.listdir("data/" + d) if f.endswith(".csv")]

        for f in files:
            input_file = f"data/{d}/{f}"
            output_file = f"data/{d}/annotated_{f}"
            annotate(input_file, output_file)