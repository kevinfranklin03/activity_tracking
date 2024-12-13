import pandas as pd

# Path to the dataset
DATA_PATH = ".venv\data\processed_dataset.csv"
OUTPUT_PATH = ".venv\data\cleaned_dataset.csv"

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Drop the last three columns: 'activity', 'subject', 'segment'
columns_to_drop = ["activity", "subject", "segment"]
data_cleaned = data.drop(columns=columns_to_drop)

# Save the cleaned dataset to a new file
data_cleaned.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned dataset saved to {OUTPUT_PATH}")
