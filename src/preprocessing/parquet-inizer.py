import os
import pandas as pd
import pdfplumber
import re


def clean_text(text):
    # Your text cleaning logic here
    # Example: remove multiple newlines, headers, footers
    cleaned_text = re.sub(r'\n\s*\n', '\n', text)
    return cleaned_text


def process_pdf(file_path):
    print(f"  Attempting to open and extract text from: {file_path}") # More detailed progress
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return clean_text(text)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    # Construct absolute paths relative to the script's location
    script_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'raw'))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed'))

    all_data = []
    processed_count = 0 # Initialize a counter for processed documents

    print(f"Starting PDF processing from: {root_dir}")
    print(f"Output Parquet file will be saved to: {output_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pdf'):
                file_path = os.path.join(dirpath, filename)
                print(f"[{processed_count + 1}] Processing: {file_path}") # Show current file and total processed so far
                extracted_text = process_pdf(file_path)

                if extracted_text:
                    # Determine the source from the file path
                    source = os.path.basename(os.path.dirname(file_path))

                    # Store the data
                    all_data.append({
                        'filename': filename,
                        'source': source,
                        'text': extracted_text,
                        'file_path': file_path
                    })
                    processed_count += 1 # Increment counter for successfully processed files
                else:
                    print(f"  Skipping {filename} due to extraction error or no text found.")

    # Create a DataFrame and save it to a Parquet file
    df = pd.DataFrame(all_data)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(os.path.join(output_dir, 'all_documents.parquet'), index=False)
    print(f"Successfully processed {len(df)} documents and saved to a Parquet file.")
    if len(df) == 0:
        print("No documents were processed. Please ensure your 'data/raw' directory exists and contains PDF files.")


if __name__ == '__main__':
    main()