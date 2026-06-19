import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_data(file_path):
    df = pd.read_csv(file_path)
    print("Data Loaded Successfully")
    return df


def transform_data(df):

    # Remove missing values
    df = df.dropna()

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Feature Scaling
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Data Transformation Completed")
    return df


def load_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


def main():

    input_file = "sample_data.csv"
    output_file = "processed_data.csv"

    data = extract_data(input_file)
    transformed_data = transform_data(data)
    load_data(transformed_data, output_file)

    print("\nETL Pipeline Executed Successfully")

if __name__ == "__main__":
    main()
