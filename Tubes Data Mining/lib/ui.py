from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLineEdit, QLabel, QFileDialog, 
                            QTextEdit, QListWidget, QMessageBox, QProgressBar)
from PyQt6.QtCore import Qt
import os

class SearchUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Search System")
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
        self.clear_files_btn = QPushButton("Clear Files")
        file_buttons.addWidget(self.add_file_btn)
        file_buttons.addWidget(self.clear_files_btn)
        file_buttons.addStretch()
        file_layout.addLayout(file_buttons)

        layout.addLayout(file_layout)

        # Search area
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query...")
        self.search_btn = QPushButton("Search")
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
        self.clear_files_btn.clicked.connect(self.clear_files)
        self.search_btn.clicked.connect(self.search)
        self.search_input.returnPressed.connect(self.search)

        # Store file paths
        self.file_paths = []

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
        self.results_text.append("Search Results:\n")
        self.results_text.append("-" * 50 + "\n")
        
        for doc, score in results:
            self.results_text.append(f"Document: {os.path.basename(self.file_paths[doc['id']])}")
            self.results_text.append(f"Similarity Score: {score:.4f}\n")
            self.results_text.append("-" * 50 + "\n")

    def update_progress(self, value, maximum=100):
        self.progress.setMaximum(maximum)
        self.progress.setValue(value)
        if value >= maximum:
            self.progress.hide()
        else:
            self.progress.show()
