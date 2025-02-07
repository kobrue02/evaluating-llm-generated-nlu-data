import pandas as pd
import csv


input_csv = "your_input_file.csv"
output_csv = "annotated_output.csv"


df = pd.read_csv(input_csv)

if "text" not in df.columns:
    raise ValueError("The CSV file must contain a 'text' column.")

# Create new columns for annotations
df["Linguistic Naturalness"] = None
df["Semantic Appropriateness"] = None
df["Overall Quality"] = None

# Iterate over the 'text' column for annotation
for index, row in df.iterrows():
    print("\nSample:", row["text"])
    
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
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"\nAnnotation completed. Results saved to {output_csv}.")
