import csv
import re
import PyPDF2
from docx import Document
import os
from stem import Stemmer  
from vsm import VSM
from export import export_to_word
import sys
from PyQt6.QtWidgets import QApplication
from ui import SearchUI

def clean_text(text):
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    return text

def read_file_content(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return clean_text(text)
            
    elif file_extension == '.docx':
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return clean_text(text)
        
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return clean_text(file.read())
    
    else:
        raise ValueError("Unsupported file format")

class SearchApplication:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = SearchUI()
        self.stemmer = Stemmer()
        self.vsm = VSM(self.stemmer)
        
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        self.window.search_btn.clicked.connect(self.search_documents)
        self.window.search_input.returnPressed.connect(self.search_documents)
        
    def run(self):
        self.window.show()
        return self.app.exec()
        
    def search_documents(self):
        if not self.window.file_paths:
            self.window.show_error("Masukkan dokumen terlebih dahulu!")
            return
            
        query = self.window.search_input.text().strip()
        if not query:
            self.window.show_error("Masukkan query yang diinginkan!")
            return
            
        try:
            self.vsm = VSM(self.stemmer)
            self.window.progress.show()
            total_files = len(self.window.file_paths)
            
            for i, file_path in enumerate(self.window.file_paths):
                text = read_file_content(file_path)
                self.vsm.add_document(i, text)
                self.window.update_progress(i + 1, total_files)
                
            self.vsm.calculate_weights()
            
            results = self.vsm.search(query)
            
            self.window.show_results(results)
            
            for doc_id in range(len(self.window.file_paths)):
                doc_path = self.window.file_paths[doc_id]
                text = read_file_content(doc_path)
                stemmed_text, steps = self.stemmer.stem_text(text)
                
                vsm_data = {
                    'terms': self.vsm.terms,
                    'documents': self.vsm.documents,
                    'term_doc_freq': self.vsm.term_doc_freq,
                    'doc_vectors': self.vsm.doc_vectors,
                    'tf_idf_vectors': self.vsm.tf_idf_vectors,
                    'idf': self.vsm.idf,
                    'query': query,
                    'search_results': results
                }
                
                export_to_word(text, stemmed_text, steps, doc_path, 
                             vsm_data, self.stemmer, self.window.file_paths)
                
        except Exception as e:
            self.window.show_error(f"An error occurred: {str(e)}")
        finally:
            self.window.progress.hide()

if __name__ == "__main__":
    app = SearchApplication()
    sys.exit(app.run())
