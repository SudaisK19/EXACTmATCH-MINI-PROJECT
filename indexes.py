import os
import sys
import re
import string
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QStackedWidget, QFrame
)
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt, QSize

nltk.download('punkt')

#######################################
# IR Functions
#######################################

def tokenize(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ''.join(ch for ch in text if not ch.isdigit())
    return nltk.word_tokenize(text.lower())

def get_stopwords(stopwords_file='Stopword-List.txt'):
    stopwords = set()
    if os.path.exists(stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = {w.strip() for w in f if w.strip()}
    return stopwords

def preprocess_text(text, stopwords):
    tokens = tokenize(text)
    stemmer = PorterStemmer()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return [stemmer.stem(t) for t in tokens]

def parse_document(contents):
    return contents.lower()

def read_documents(folder_path):
    documents = {}
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found; creating it.")
        os.makedirs(folder_path)
        return documents
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='cp1252') as f:
                content = f.read()
            doc_text = parse_document(content)
            doc_id = re.sub(r'\D', "", filename) or filename
            doc_id = int(doc_id) if doc_id.isdigit() else doc_id
            documents[doc_id] = doc_text
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return documents

def preprocess_documents(documents, stopwords):
    preprocessed = {}
    for doc_id, text in documents.items():
        preprocessed[doc_id] = preprocess_text(text, stopwords)
    return preprocessed

def generate_inverted_index(preprocessed_docs):
    index = defaultdict(set)
    for doc_id, tokens in preprocessed_docs.items():
        for token in tokens:
            index[token].add(doc_id)
    return {term: sorted(docs) for term, docs in index.items()}

def generate_positional_index(preprocessed_docs):
    pos_index = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in preprocessed_docs.items():
        for pos, token in enumerate(tokens):
            pos_index[token][doc_id].append(pos)
    return {term: dict(postings) for term, postings in pos_index.items()}

def boolean_query(query, inverted_index, stopwords):
    tokens = nltk.word_tokenize(query)
    stemmer = PorterStemmer()
    result = None
    op = None
    for token in tokens:
        t_upper = token.upper()
        if t_upper in ("AND", "OR", "NOT"):
            op = t_upper
        else:
            term = stemmer.stem(token.lower())
            posting = set(inverted_index.get(term, []))
            if result is None:
                result = posting
            else:
                if op == "AND":
                    result = result.intersection(posting)
                elif op == "OR":
                    result = result.union(posting)
                elif op == "NOT":
                    result = result.difference(posting)
                else:
                    result = result.intersection(posting)
            op = None
    final_set = result if result else set()
    return (final_set, None)

def proximity_query(query, positional_index, stopwords):
    tokens = nltk.word_tokenize(query)
    k = None
    term1, term2 = None, None
    if "/" in query:
        parts = query.split("/")
        try:
            k = int(parts[1])
        except:
            print("Invalid proximity number.")
            return (set(), {})
        left_part = parts[0].strip()
        terms = nltk.word_tokenize(left_part)
        if len(terms) < 2:
            print("Proximity query must have two terms.")
            return (set(), {})
        term1, term2 = terms[0], terms[1]
    else:
        if len(tokens) < 3:
            print("Proximity query must have two terms + distance.")
            return (set(), {})
        term1, term2 = tokens[0], tokens[1]
        try:
            k = int(tokens[2])
        except:
            print("Invalid proximity distance.")
            return (set(), {})
    stemmer = PorterStemmer()
    t1 = stemmer.stem(term1.lower())
    t2 = stemmer.stem(term2.lower())

    postings1 = positional_index.get(t1, {})
    postings2 = positional_index.get(t2, {})

    results = set()
    matched_positions = defaultdict(lambda: defaultdict(set))
    common_docs = set(postings1.keys()).intersection(postings2.keys())
    for doc_id in common_docs:
        pos1 = postings1[doc_id]
        pos2 = postings2[doc_id]
        matched_p1 = set()
        matched_p2 = set()
        for p1 in pos1:
            for p2 in pos2:
                if abs(p1 - p2) <= k and p1 != p2:
                    matched_p1.add(p1)
                    matched_p2.add(p2)
        if matched_p1 and matched_p2:
            results.add(doc_id)
            matched_positions[doc_id][t1].update(matched_p1)
            matched_positions[doc_id][t2].update(matched_p2)
    return (results, matched_positions)

def process_query(query, inverted_index, positional_index, stopwords):
    if "/" in query:
        return proximity_query(query, positional_index, stopwords)
    else:
        return boolean_query(query, inverted_index, stopwords)

#######################################
# HomePage - SERP
#######################################

class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(20,20,20,20)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_layout.setSpacing(10)
        self.results_layout.setContentsMargins(0,0,0,0)

        self.scroll_area.setWidget(self.results_container)

    def update_results(self, result_docs, documents):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not result_docs:
            no_label = QLabel("No matching documents found.")
            no_label.setStyleSheet("color: #888; font-size:14px;")
            self.results_layout.addWidget(no_label)
        else:
            for doc_id in sorted(result_docs):
                snippet = documents.get(doc_id, "")
                snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet

                result_widget = QWidget()
                layout = QVBoxLayout(result_widget)
                layout.setSpacing(2)
                layout.setContentsMargins(0,0,0,10)

                title_label = QLabel(f"Document {doc_id}")
                title_label.setStyleSheet("color: #2619AF; font-size:18px; font-weight:bold;")

                url_label = QLabel(f"<span style='color: #078307; font-size:13px;'>Document_{doc_id}.txt</span>")
                snippet_label = QLabel(snippet)
                snippet_label.setStyleSheet("color: black; font-size:14px;")
                snippet_label.setWordWrap(True)

                layout.addWidget(title_label)
                layout.addWidget(url_label)
                layout.addWidget(snippet_label)

                self.results_layout.addWidget(result_widget)

#######################################
# PostingsPage
#######################################

class PostingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5,5,5,5)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setSpacing(5)
        self.container_layout.setContentsMargins(5,5,5,5)
        self.scroll_area.setWidget(self.container)

    def clear_display(self):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def show_index(self, query, query_docs, matched_positions, inverted_index, positional_index):
        self.clear_display()
        from nltk import word_tokenize
        from nltk.stem import PorterStemmer

        is_proximity = ("/" in query)
        tokens = word_tokenize(query)
        stemmer = PorterStemmer()

        if is_proximity:
            parts = query.split("/")
            left_part = parts[0].strip()
            terms = word_tokenize(left_part)
        else:
            terms = [t for t in tokens if t.upper() not in ("AND","OR","NOT")]

        term_box_size = (150, 150)
        arrow_box_size = (50, 150)
        docs_box_size = (600, 150)

        for term in terms:
            t_stem = stemmer.stem(term.lower())

            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)

            term_box = QFrame()
            term_box.setFixedSize(*term_box_size)
            term_box.setStyleSheet("border:1px solid #ccc; background-color:#fafafa;")
            term_layout = QVBoxLayout(term_box)
            term_layout.setContentsMargins(5,5,5,5)
            term_label = QLabel(f"<b>{term}</b>")
            term_label.setStyleSheet("font-size:14px; color:#333;")
            term_layout.addWidget(term_label, alignment=Qt.AlignCenter)

            arrow_box = QFrame()
            arrow_box.setFixedSize(*arrow_box_size)
            arrow_box.setStyleSheet("border:1px solid #ccc; background-color:#fafafa;")
            arrow_layout = QVBoxLayout(arrow_box)
            arrow_layout.setContentsMargins(5,5,5,5)
            arrow_label = QLabel("→")
            arrow_label.setStyleSheet("font-size:16px; color:#777;")
            arrow_layout.addWidget(arrow_label, alignment=Qt.AlignCenter)

            docs_box = QFrame()
            docs_box.setFixedSize(*docs_box_size)
            docs_box.setStyleSheet("border:1px solid #ccc; background-color:#fafafa;")
            docs_layout = QVBoxLayout(docs_box)
            docs_layout.setContentsMargins(0,0,0,0)
            docs_layout.setSpacing(0)

            doc_scroll = QScrollArea()
            doc_scroll.setWidgetResizable(True)
            doc_scroll.setStyleSheet("border:none; background-color:transparent;")

            doc_content = QWidget()
            doc_content_layout = QVBoxLayout(doc_content)
            doc_content_layout.setContentsMargins(5,5,5,5)

            doc_info_label = QLabel()
            doc_info_label.setStyleSheet("font-size:13px; color:#444;")
            doc_info_label.setTextFormat(Qt.RichText)
            doc_info_label.setWordWrap(True)

            doc_content_layout.addWidget(doc_info_label)
            doc_scroll.setWidget(doc_content)
            docs_layout.addWidget(doc_scroll)

            if is_proximity and matched_positions:
                postings = positional_index.get(t_stem, {})
                if postings:
                    all_doc_ids = sorted(postings.keys())
                    doc_ids_str = []
                    for d in all_doc_ids:
                        if d in query_docs:
                            doc_ids_str.append(f"<span style='color:red;'>{d}</span>")
                        else:
                            doc_ids_str.append(str(d))
                    doc_ids_line = ", ".join(doc_ids_str)
                    info_text = f"<b>Doc IDs:</b> {doc_ids_line}<br>"

                    for d in sorted(query_docs):
                        if d not in postings:
                            continue
                        positions = postings[d]
                        if d in matched_positions and t_stem in matched_positions[d]:
                            highlight_set = matched_positions[d][t_stem]
                        else:
                            highlight_set = set()
                        pos_list = []
                        for p in positions:
                            if p in highlight_set:
                                pos_list.append(f"<span style='color:red;'>{p}</span>")
                            else:
                                pos_list.append(str(p))
                        pos_line = ", ".join(pos_list)
                        info_text += f"<span style='color:red;'>Doc {d}</span>: [{pos_line}]<br>"
                    doc_info_label.setText(info_text)
                else:
                    doc_info_label.setText("<i>No postings for this term.</i>")
            else:
                docs = inverted_index.get(t_stem, [])
                if docs:
                    parts_list = []
                    for d in docs:
                        if d in query_docs:
                            parts_list.append(f"<span style='color:red;'>{d}</span>")
                        else:
                            parts_list.append(str(d))
                    joined = ", ".join(parts_list)
                    doc_info_label.setText(joined)
                else:
                    doc_info_label.setText("<i>No documents for this term.</i>")

            row_layout.addWidget(term_box)
            row_layout.addWidget(arrow_box)
            row_layout.addWidget(docs_box)

            row_container = QWidget()
            row_container.setLayout(row_layout)
            self.container_layout.addWidget(row_container)

#######################################
# AboutPage - bigger text, details, stopwords
#######################################

class AboutPage(QWidget):
    def __init__(self, parent=None, stopwords_list=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0,0,0,0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(20,20,20,20)
        container_layout.setSpacing(10)

        heading_html = """
        <h1 style='text-align:center; font-family: sans-serif; font-size:48px;'>
          <span style='color:#4285F4;'>E</span>
          <span style='color:#EA4335;'>x</span>
          <span style='color:#FBBC05;'>a</span>
          <span style='color:#4285F4;'>c</span>
          <span style='color:#34A853;'>t</span>
          <span style='color:#EA4335;'>M</span>
          <span style='color:#FBBC05;'>a</span>
          <span style='color:#4285F4;'>t</span>
          <span style='color:#34A853;'>c</span>
          <span style='color:#EA4335;'>h</span>
        </h1>
        """
        heading = QLabel(heading_html)
        heading.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(heading)

        about_text = """<p style='font-size:16px; line-height:1.5;'>
        <strong>Teacher's Task Details</strong><br>
        You have been tasked with implementing a simple Boolean Information Retrieval system, 
        which includes building both an Inverted Index and a Positional Index for a set of documents (abstracts). 
        This system must handle Boolean queries (with AND, OR, NOT operators) as well as Proximity queries 
        (using the format <em>term1 term2 / k</em>, meaning the two terms occur within k words of each other).<br><br>

        <strong>Implementation Summary</strong><br>
        1. <em>Preprocessing</em>: We remove punctuation and digits, convert text to lowercase, remove stopwords, 
        and apply stemming (PorterStemmer).<br>
        2. <em>Inverted Index</em>: For each unique term, store a sorted list of document IDs in which it appears.<br>
        3. <em>Positional Index</em>: For each unique term, store the positions of that term in each document 
        (used for Proximity queries).<br>
        4. <em>Query Processing</em>: 
           - Boolean queries are parsed for AND/OR/NOT logic. 
           - Proximity queries look for terms within k tokens in each document.<br>
        5. <em>GUI</em>: The interface has a Home page for results, a Postings page for seeing the index details, 
        and an About page for assignment info and stopwords.<br><br>

        This approach allows you to quickly find which documents contain certain terms (Inverted Index), 
        and also how close those terms appear to each other (Positional Index). 
        It is a simplified demonstration of a typical Boolean IR model with proximity support.<br><br>
        </p>
        """
        about_label = QLabel(about_text)
        about_label.setWordWrap(True)
        container_layout.addWidget(about_label)

        # Show stopwords with bigger heading
        stopwords_heading = QLabel("<strong style='font-size:16px;'>Stop Words:</strong>")
        container_layout.addWidget(stopwords_heading)

        if stopwords_list is None:
            stopwords_list = []
        sw_sorted = sorted(stopwords_list)
        sw_text = ", ".join(sw_sorted)
        sw_label = QLabel(f"<p style='font-size:14px; line-height:1.4;'>{sw_text}</p>")
        sw_label.setWordWrap(True)
        container_layout.addWidget(sw_label)

        container_layout.addStretch()
        scroll_area.setWidget(container)
        self.setLayout(main_layout)

#######################################
# ExactMatchGUI
#######################################

class ExactMatchGUI(QMainWindow):
    def __init__(self, inverted_index, positional_index, stopwords, documents):
        super().__init__()
        self.inverted_index = inverted_index
        self.positional_index = positional_index
        self.stopwords = stopwords
        self.documents = documents
        self.setWindowTitle("ExactMatch - BoolModel Search Engine")
        self.setGeometry(100, 100, 1000, 700)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0,0,0,0)

        # 1) Big brand heading at top (multi-colored)
        brand_html = """
        <span style='font-size:60px;'>
          <span style='color:#4285F4;'>E</span>
          <span style='color:#EA4335;'>x</span>
          <span style='color:#FBBC05;'>a</span>
          <span style='color:#4285F4;'>c</span>
          <span style='color:#34A853;'>t</span>
          <span style='color:#EA4335;'>M</span>
          <span style='color:#FBBC05;'>a</span>
          <span style='color:#4285F4;'>t</span>
          <span style='color:#34A853;'>c</span>
          <span style='color:#EA4335;'>h</span>
        </span>
        """
        brand_label = QLabel(brand_html)
        brand_label.setAlignment(Qt.AlignCenter)

        brand_layout = QHBoxLayout()
        brand_layout.addStretch()
        brand_layout.addWidget(brand_label)
        brand_layout.addStretch()

        main_layout.addLayout(brand_layout)

        # 2) Search row
        search_layout = QHBoxLayout()
        search_layout.setSpacing(5)
        search_layout.setContentsMargins(20, 10, 20, 10)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter your query (Boolean or Proximity)...")
        self.search_input.setFixedHeight(40)
        self.search_input.setStyleSheet("font-size:16px; padding:8px; border:1px solid #ccc; border-radius:8px;")

        self.search_button = QPushButton()
        self.search_button.setFixedSize(40, 40)
        icon_pixmap = QPixmap("magnify.png")
        if not icon_pixmap.isNull():
            icon_pixmap = icon_pixmap.scaled(24,24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.search_button.setIcon(QIcon(icon_pixmap))
        else:
            self.search_button.setText("Go")
        self.search_button.setStyleSheet("background-color: #f8f8f8; border:1px solid #ccc; border-radius:8px;")
        self.search_button.clicked.connect(self.handle_search)

        search_layout.addStretch()
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addStretch()

        main_layout.addLayout(search_layout)

        # 3) Navigation row (Home, Postings, About) in grey
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)
        nav_layout.setContentsMargins(10, 0, 10, 10)

        self.home_button = QPushButton("Home")
        self.home_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
                border-bottom: 2px solid crimson;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.home_button.clicked.connect(self.show_home_page)

        self.postings_button = QPushButton("Postings")
        self.postings_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.postings_button.clicked.connect(self.show_postings_page)

        self.about_button = QPushButton("About")
        self.about_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.about_button.clicked.connect(self.show_about_page)

        nav_layout.addWidget(self.home_button)
        nav_layout.addWidget(self.postings_button)
        nav_layout.addWidget(self.about_button)
        nav_layout.addStretch()

        main_layout.addLayout(nav_layout)

        # 4) Stacked pages
        self.pages = QStackedWidget()
        self.home_page = HomePage()         # index 0
        self.postings_page = PostingsPage() # index 1
        self.about_page = AboutPage(stopwords_list=self.stopwords) # index 2

        self.pages.addWidget(self.home_page)
        self.pages.addWidget(self.postings_page)
        self.pages.addWidget(self.about_page)
        main_layout.addWidget(self.pages)

    def show_home_page(self):
        self.home_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
                border-bottom: 2px solid crimson;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.postings_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.about_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.pages.setCurrentIndex(0)

    def show_postings_page(self):
        self.home_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.postings_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
                border-bottom: 2px solid crimson;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.about_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.pages.setCurrentIndex(1)

    def show_about_page(self):
        self.home_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.postings_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.about_button.setStyleSheet("""
            QPushButton {
                color: #666; font-size:14px; font-weight:bold; border:none;
                border-bottom: 2px solid crimson;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)
        self.pages.setCurrentIndex(2)

    def handle_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        final_docs, matched_positions = process_query(query, self.inverted_index, self.positional_index, self.stopwords)
        self.home_page.update_results(final_docs, self.documents)
        self.postings_page.show_index(query, final_docs, matched_positions, self.inverted_index, self.positional_index)
        self.show_home_page()

#######################################
# Main
#######################################

if __name__ == "__main__":
    if len(sys.argv) > 1:
        docs_folder = sys.argv[1]
    else:
        docs_folder = "documents"

    stopwords = get_stopwords()
    documents = read_documents(docs_folder)
    preprocessed_docs = preprocess_documents(documents, stopwords)
    inverted_index = generate_inverted_index(preprocessed_docs)
    positional_index = generate_positional_index(preprocessed_docs)

    print("Indexes built. Launching ExactMatch GUI...")

    app = QApplication(sys.argv)
    window = ExactMatchGUI(inverted_index, positional_index, stopwords, documents)
    window.show()
    sys.exit(app.exec_())
