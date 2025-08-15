import pandas as pd

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.where(pd.notnull(df), '')
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess("..data/mail_data.csv", "../data/clean_mail_data.csv")
