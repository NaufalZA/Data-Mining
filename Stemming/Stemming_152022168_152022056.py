import csv
import re
from tkinter import filedialog
import tkinter as tk
import PyPDF2
from docx import Document
import os

class Stemmer:
    def __init__(self):
        self.kamus = self.load_kamus()
        self.stopwords = self.load_stopwords()

    def load_kamus(self):
        with open('documents/Kamus.txt', 'r') as file:
            return set(word.strip().lower() for word in file)

    def load_stopwords(self):
        stopwords = set()
        with open('documents/Stopword.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                stopwords.add(row[0].lower())
        return stopwords

    def is_valid_word(self, word):
        return bool(re.match('^[a-zA-Z]+$', word))

    def remove_stopwords(self, text):
        words = text.lower().split()
        return ' '.join([word for word in words if word not in self.stopwords])

    def check_kamus(self, word):
        return word.lower() in self.kamus

    def remove_inflection_suffixes(self, word):
        # Remove -lah, -kah, -tah, -pun
        if word.endswith(('lah', 'kah', 'tah', 'pun')):
            word = re.sub(r'(lah|kah|tah|pun)$', '', word)
            
        # Remove -ku, -mu, -nya
        if word.endswith(('ku', 'mu', 'nya')):
            word = re.sub(r'(ku|mu|nya)$', '', word)
            
        return word

    def remove_derivation_suffixes(self, word):
        if word.endswith(('i', 'an', 'kan')):
            original = word
            
            # Remove -i, -an, or -kan
            if word.endswith('kan'):
                word = word[:-3]
            elif word.endswith(('i', 'an')):
                word = word[:-2]

            # Special case for -an where last letter is k
            if original.endswith('an') and word.endswith('k'):
                word = word[:-1]
                if self.check_kamus(word):
                    return word
                word = word + 'k'

            # Check if result exists in dictionary
            if self.check_kamus(word):
                return word
                
            # If not found, restore the original suffix
            return original
            
        return word

    def is_forbidden_combination(self, prefix, suffix):
        forbidden = {
            'be': ['i'],
            'di': ['an'],
            'ke': ['i', 'kan'],
            'me': ['an'],
            'se': ['i', 'kan']
        }
        return prefix in forbidden and suffix in forbidden[prefix]

    def remove_prefix(self, word, iteration=1):
        if iteration > 3:
            return word

        if word[:2] in ['di', 'ke', 'se']:
            prefix = word[:2]
            stemmed = word[2:]
            
        elif word.startswith(('ter', 'bel')):
            prefix = word[:3]
            stemmed = word[3:]
            
        elif word.startswith(('me', 'pe', 'be')):
            if len(word) > 3:
                if word[2] == 'r' and word[3] in ['a','i','u','e','o']:
                    prefix = word[:3]
                    stemmed = word[3:]
                else:
                    prefix = word[:2]
                    stemmed = word[2:]
            else:
                prefix = word[:2]
                stemmed = word[2:]
        else:
            return word

        # Check if result exists in dictionary
        if self.check_kamus(stemmed):
            return stemmed
            
        # Try next iteration
        return self.remove_prefix(word, iteration + 1)

    def stem_word(self, word):
        if not self.is_valid_word(word):
            return word
            
        # Step 1: Check in dictionary
        if self.check_kamus(word):
            return word

        # Step 2: Remove inflection suffixes
        word = self.remove_inflection_suffixes(word)
        if self.check_kamus(word):
            return word

        # Step 3: Remove derivation suffixes
        word = self.remove_derivation_suffixes(word)
        if self.check_kamus(word):
            return word

        # Step 4: Remove prefixes
        word = self.remove_prefix(word)
        
        return word

    def stem_text(self, text):
        # Remove stopwords first
        text = self.remove_stopwords(text)
        
        # Split into words and stem each word
        words = text.split()
        stemmed_words = [self.stem_word(word) for word in words]
        
        return ' '.join(stemmed_words)

def read_file_content(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
            
    elif file_extension == '.docx':
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
        
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    else:
        raise ValueError("Unsupported file format")

def export_to_word(original_text, stemmed_text, input_file_path):
    # Get original filename without extension
    original_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    output_filename = f"results/Stemmed_{original_filename}.docx"
    
    # Create new Word document
    doc = Document()
    doc.add_heading('Stemming Results', 0)
    
    # Add original text section
    doc.add_heading('Original Text:', level=1)
    doc.add_paragraph(original_text)
    
    # Add stemmed text section
    doc.add_heading('Stemmed Text:', level=1)
    doc.add_paragraph(stemmed_text)
    
    # Save document
    doc.save(output_filename)
    return output_filename

if __name__ == "__main__":
    stemmer = Stemmer()
    
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select File",
        filetypes=[
            ("All supported files", "*.txt;*.pdf;*.docx"),
            ("Text files", "*.txt"),
            ("PDF files", "*.pdf"),
            ("Word files", "*.docx"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        try:
            text = read_file_content(file_path)
            stemmed_text = stemmer.stem_text(text)
            
            print(f"\nOriginal text from file:")
            print(text)
            print(f"\nStemmed text:")
            print(stemmed_text)
            
            # Export results to Word document
            output_file = export_to_word(text, stemmed_text, file_path)
            print(f"\nResults exported to: {output_file}")
            
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        print("No file selected")
