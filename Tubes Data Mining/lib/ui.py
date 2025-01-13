from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLineEdit, QLabel, QFileDialog, 
                            QTextEdit, QListWidget, QMessageBox, QProgressBar)
from PyQt6.QtCore import Qt
import os

class SearchUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("House of Algorithm")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # File selection area
        file_layout = QHBoxLayout()
        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)

        file_buttons = QVBoxLayout()
        self.add_file_btn = QPushButton("Add Files")
        self.remove_file_btn = QPushButton("Remove File")
        self.clear_files_btn = QPushButton("Clear Files")
        file_buttons.addWidget(self.add_file_btn)
        file_buttons.addWidget(self.remove_file_btn)
        file_buttons.addWidget(self.clear_files_btn)
        file_buttons.addStretch()
        file_layout.addLayout(file_buttons)

        layout.addLayout(file_layout)

        # Search area
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Masukkan Query...")
        self.search_btn = QPushButton("Cari")
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_btn)
        layout.addLayout(search_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)

        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        # Connect signals
        self.add_file_btn.clicked.connect(self.add_files)
        self.remove_file_btn.clicked.connect(self.remove_selected_file)
        self.clear_files_btn.clicked.connect(self.clear_files)
        self.search_btn.clicked.connect(self.search)
        self.search_input.returnPressed.connect(self.search)
        self.file_list.itemDoubleClicked.connect(self.open_file)
        self.results_text.mouseDoubleClickEvent = self.handle_results_double_click

        # Store file paths
        self.file_paths = []
        
        # Load files from text directory
        self.load_text_directory()

    def load_text_directory(self):
        text_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'text')
        if os.path.exists(text_dir):
            for filename in os.listdir(text_dir):
                file_path = os.path.join(text_dir, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.txt', '.pdf', '.docx')):
                    if file_path not in self.file_paths:
                        self.file_paths.append(file_path)
                        self.file_list.addItem(os.path.basename(file_path))

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents",
            "",
            "Documents (*.txt *.pdf *.docx)"
        )
        
        if files:
            for file_path in files:
                if file_path not in self.file_paths:
                    self.file_paths.append(file_path)
                    self.file_list.addItem(os.path.basename(file_path))

    def remove_selected_file(self):
        current_item = self.file_list.currentItem()
        if current_item:
            file_name = current_item.text()
            for i, path in enumerate(self.file_paths):
                if os.path.basename(path) == file_name:
                    self.file_paths.pop(i)
                    break
            self.file_list.takeItem(self.file_list.row(current_item))

    def clear_files(self):
        self.file_paths.clear()
        self.file_list.clear()
        self.results_text.clear()

    def search(self):
        # This will be implemented in main.py
        pass

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_results(self, results):
        self.results_text.clear()
        self.results_text.append("Documents have been processed and exported. Results files:\n")
        self.results_text.append("-" * 50 + "\n")
        
        for doc, score in results:
            original_file = os.path.basename(self.file_paths[doc['id']])
            result_file = f"Hasil_{os.path.splitext(original_file)[0]}.docx"
            result_path = os.path.join('results', result_file)
            
            self.results_text.append(f"Dokumen: {original_file}")
            self.results_text.append(f"Processing: {result_file}")
            self.results_text.append(f"Similarity Score: {score:.4f}\n")
            self.results_text.append("-" * 50 + "\n")

    def update_progress(self, value, maximum=100):
        self.progress.setMaximum(maximum)
        self.progress.setValue(value)
        if value >= maximum:
            self.progress.hide()
        else:
            self.progress.show()

    def open_file(self, item):
        file_name = item.text()
        for path in self.file_paths:
            if os.path.basename(path) == file_name:
                os.startfile(path)
                break

    def handle_results_double_click(self, event):
        cursor = self.results_text.cursorForPosition(event.pos())
        text = cursor.block().text()
        if "Result file:" in text:
            result_file = text.split("Result file:")[1].strip()
            result_path = os.path.join('results', result_file)
            if os.path.exists(result_path):
                os.startfile(result_path)
        elif "Document:" in text:
            file_name = text.split("Document:")[1].strip()
            for path in self.file_paths:
                if os.path.basename(path) == file_name:
                    os.startfile(path)
                    break
