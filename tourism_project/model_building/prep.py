import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 1. Define paths
    data_path = "tourism_project/data/tourism.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please check the file path.")

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # 2. Remove unnecessary columns
    # 'CustomerID' is a unique identifier and should not be used for model prediction
    columns_to_drop = ['CustomerID']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 3. Handle missing values (simple baseline: drop missing rows)
    df = df.dropna()

    # 4. Separate target variable and features
    target_col = 'ProdTaken'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Save split files to local root directory (for GitHub Actions artifact uploading)
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print("Data preparation complete!")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print("Saved split files: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

if __name__ == "__main__":
    main()
