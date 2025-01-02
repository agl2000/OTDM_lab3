import pandas as pd

# Load the CSV file into a dataframe
data = pd.read_csv("crop_yield_data.csv")

# Display summary statistics for each column
summary_statistics = data.describe()

# Print the summary statistics
print(summary_statistics)