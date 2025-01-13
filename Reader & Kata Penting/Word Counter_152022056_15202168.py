import os
import re
from collections import Counter
import docx 
import fitz  
import pandas as pd 

NRP_1 = "152022056"
NAMA_1 = "Emelsha Viadra"
NRP_2 = "152022168"
NAMA_2 = "Naufal Zaidan"
STOPWORD_FILE = r"stopwordbahasa.csv"

def load_stopwords(filepath):
    """Membaca file stopword dari CSV dan mengembalikan daftar stopword."""
    stopwords = pd.read_csv(filepath, header=None)[0].tolist()
    return set(stopwords)

def read_txt(filepath):
    """Membaca isi file .txt."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def read_docx(filepath):
    """Membaca isi file .docx."""
    doc = docx.Document(filepath)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(filepath):
    """Membaca isi file .pdf."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def tokenize(text):
    """Melakukan tokenisasi teks (mengubah teks menjadi daftar kata)."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def filter_tokens(tokens, stopwords):
    """Menghapus kata-kata tidak penting dari daftar token."""
    return [token for token in tokens if token not in stopwords]

def process_document(filepath, stopwords):
    """Memproses file berdasarkan formatnya, menghitung kata penting."""
    if filepath.endswith(".txt"):
        content = read_txt(filepath)
    elif filepath.endswith(".docx"):
        content = read_docx(filepath)
    elif filepath.endswith(".pdf"):
        content = read_pdf(filepath)
    else:
        raise ValueError(f"Format file tidak didukung: {filepath}")

    tokens = tokenize(content)
    filtered_tokens = filter_tokens(tokens, stopwords)
    word_counts = Counter(filtered_tokens)
    return word_counts

def main():
    """Fungsi utama untuk meminta input dari pengguna dan memproses dokumen di direktori yang dipilih."""
    print(f"NRP: {NRP_1}")
    print(f"NAMA: {NAMA_1}")
    print(f"NRP: {NRP_2}")
    print(f"NAMA: {NAMA_2}\n")
    
    directory = input("Masukkan path direktori tempat file yang ingin diproses: ")
    stopwords = load_stopwords(STOPWORD_FILE)
    print("Memproses dokumen di direktori:", directory)
    
    if not os.path.exists(directory):
        print(f"Direktori {directory} tidak ditemukan.")
        return

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue  # Lewati jika bukan file
        if filepath.endswith((".txt", ".docx", ".pdf")):
            print(f"\nMembaca file: {filename}")
            try:
                word_counts = process_document(filepath, stopwords)
                print(f"Kata penting di {filename}:")
                for word, count in word_counts.items():
                    print(f"- Kata '{word}' sebanyak {count} kata")
            except Exception as e:
                print(f"Error memproses file {filename}: {e}")

if __name__ == "__main__":
    main()
