import os
from docx import Document
import PyPDF2

def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Gagal membaca file TXT: {str(e)}"

def read_docx(file_path):
    try:
        doc = Document(file_path)
        content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return content
    except Exception as e:
        return f"Gagal membaca file DOCX: {str(e)}"

def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            content = ''
            for page in pdf_reader.pages:
                content += page.extract_text()
        return content
    except Exception as e:
        return f"Gagal membaca file PDF: {str(e)}"

def read_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.txt':
        return read_txt(file_path)
    elif file_extension in ['.doc', '.docx']:
        return read_docx(file_path)
    elif file_extension == '.pdf':
        return read_pdf(file_path)
    else:
        return "Format file tidak didukung"

def hitung(text):
    if not text:
        return 0, 0
    words = text.split()
    kata = len(words)
    huruf = sum(c.isalnum() for c in text)
    return kata, huruf

def main():    
    directory_path = "documents"    
    try:        
        files = os.listdir(directory_path)        
        for file_name in files:
            if file_name.lower().endswith(('.txt', '.doc', '.docx', '.pdf')):
                file_path = os.path.join(directory_path, file_name)
                print(f"\nFile: {file_name}")
                print("-" * 50)
                content = read_document(file_path)
                jumlah_kata, jumlah_huruf = hitung(content)
                print(f"Konten: {content}")
                print(f"Jumlah kata  : {jumlah_kata}")
                print(f"Jumlah huruf : {jumlah_huruf}")
                print("-" * 50)                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
