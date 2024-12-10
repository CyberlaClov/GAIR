import pandas as pd

# Read the CSV file
df = pd.read_csv("submission_results.csv")

# Add 1 to the question_id column
df["question_id"] = df["question_id"] + 1

# Save the modified data back to the CSV file
df.to_csv("submission_results.csv", index=False)
