import csv
import re
import PyPDF2
from docx import Document
import os
from stem import Stemmer  
from vsm import VSM
from export import export_to_word

def clean_text(text):
    # Remove NULL bytes
    text = text.replace('\x00', '')
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)
    # Replace invalid XML characters
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

if __name__ == "__main__":
    stemmer = Stemmer()  # Use the imported Stemmer
    vsm = VSM(stemmer)
    
    # Replace file dialog with direct directory reading
    text_dir = "text"
    supported_extensions = ('.txt', '.pdf', '.docx')
    file_paths = []
    
    # Get all supported files from text directory
    for file in os.listdir(text_dir):
        if file.lower().endswith(supported_extensions):
            file_paths.append(os.path.join(text_dir, file))
    
    if file_paths:
        try:
            # Load documents
            for i, file_path in enumerate(file_paths):
                text = read_file_content(file_path)
                vsm.add_document(i, text)
                print(f"Dokumen {i+1}: {os.path.basename(file_path)}")
            
            # Calculate weights
            vsm.calculate_weights()
            
            while True:
                query = input("\nMasukkan kata kunci pencarian (ketik 'keluar' untuk berhenti): ")
                if query.lower() == 'keluar':
                    break
                
                results = vsm.search(query)
                print("\nHasil Pencarian:")
                print("-" * 50)
                
                # Export results for all documents
                for doc_id in range(len(file_paths)):
                    doc_path = file_paths[doc_id]
                    score = next((score for doc, score in results if doc['id'] == doc_id), 0)
                    print(f"Dokumen {doc_id + 1}: {os.path.basename(doc_path)}")
                    print(f"Nilai Kemiripan: {score:.4f}\n")
                    
                    text = read_file_content(doc_path)
                    stemmed_text, steps = stemmer.stem_text(text)
                    vsm_data = {
                        'terms': vsm.terms,
                        'documents': vsm.documents,
                        'term_doc_freq': vsm.term_doc_freq,
                        'doc_vectors': vsm.doc_vectors,
                        'tf_idf_vectors': vsm.tf_idf_vectors,
                        'idf': vsm.idf,
                        'query': query,
                        'search_results': results
                    }
                    export_to_word(text, stemmed_text, steps, doc_path, vsm_data, stemmer, file_paths)
                print("-" * 50)
                
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
    else:
        print("Tidak ada file yang ditemukan di direktori text/")
