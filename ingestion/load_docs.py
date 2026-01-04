"""
Module for loading text documents from a specified folder.

Reads `.txt` and `.md` files from the specified folder and returns a list
containing each document's filename and cleaned text content.
"""
import os

def load_text_files(folder: str = "data") -> list[dict]:
    """
    Load and preprocess documents from a folder.

    Args:
        folder (str) : Directory containing text files.

    Returns:
        list[dict]: Each item contains:
            - filename (str)
            - text (str): preprocessed document text
    """
    docs = []

    # Iterate over files in target folder
    for file in os.listdir(folder):
        if file.endswith(".txt") or file.endswith(".md"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:

                text = f.read()

                # Preprocess text: remove newlines and extra spaces
                text = text.replace("\n", " ")
                text = " ".join(text.split())
                

                docs.append({
                    "filename": file,
                    "text": text
                })

    return docs