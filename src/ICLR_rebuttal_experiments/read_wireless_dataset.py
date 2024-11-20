import pandas as pd


def read_bandit_trace(csv_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_path)

    # Display the first few rows of the DataFrame
    print("First few rows of the data:")
    print(data.head())

    # Display basic statistics about the data
    print("\nStatistics for each column:")
    print(data.describe())


if __name__ == "__main__":
    # Path to the CSV file
    csv_file_path = 'ICLR_rebuttal_data/wireless_system_dataset/trace.csv'
    read_bandit_trace(csv_file_path)
