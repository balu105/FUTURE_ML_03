import pandas as pd
import csv
from tabulate import tabulate  

def preprocess_books(input_file, output_file):
    
    with open(input_file, "r", encoding="latin1") as infile, open(output_file, "w", encoding="latin1", newline="") as outfile:
        writer = csv.writer(outfile, delimiter=";")  # Use correct delimiter
        for line in infile:
            cleaned_line = line.strip().strip('"')  # Remove unwanted surrounding quotes
            writer.writerow(cleaned_line.split('";"'))  # Split correctly

    
    df = pd.read_csv(output_file, encoding="latin1", delimiter=";", quotechar='"', on_bad_lines="skip")

   
    df = df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, errors='ignore')

  
    df.to_csv(output_file, encoding="latin1", index=False, sep=";")
    
    return df


if __name__ == "__main__":
    input_path = "/content/drive/MyDrive/A/books.csv"  # Change this to your actual file path
    output_path = "/content/drive/MyDrive/A/preprocessed_data.csv"  # Save as preprocessed_data.csv

    df = preprocess_books(input_path, output_path)

    print("Preprocessed Data (Saved as preprocessed_data.csv):")
    print(tabulate(df.head(), headers='keys', tablefmt='grid'))
    print(df.columns)