# ExactMatch - Interactive Information Retrieval System

ExactMatch is an interactive IR project that demonstrates the construction and use of inverted and positional indexes for processing Boolean and proximity queries. Built with Python and PyQt5, the system provides a user-friendly GUI that not only retrieves search results from a document collection but also visually displays the underlying index data (postings) to help understand how different queries are processed.

---

## Project Overview

The goal of ExactMatch is to implement a simplified Information Retrieval model that supports both Boolean queries (using operators like AND, OR, and NOT) and proximity queries (using a format such as `term1 term2 / k`, which returns documents where the two terms appear within *k* words of each other). The project includes:
- **Preprocessing**: Tokenization, case folding, stopword removal, and stemming (using PorterStemmer).
- **Indexing**: Construction of both an inverted index (mapping each term to the list of document IDs) and a positional index (storing term positions within documents).
- **Query Processing**: Evaluating Boolean and proximity queries against the indexes.
- **GUI**: A modern, interactive interface for entering queries and viewing both search results and index details.

---

## Features

- **Dynamic Query Processing**: Supports Boolean queries (e.g., `term1 AND term2`) and proximity queries (e.g., `term1 term2 / 5`).
- **Visual Index Postings**: Displays the list of document IDs and the positions where query terms appear, with matched positions highlighted.
- **User-Friendly Interface**: An interactive GUI built using PyQt5, styled for a clean, modern look.
- **Preprocessing Pipeline**: Includes tokenization, stopword removal, and stemming to prepare documents for indexing.

---

## Installation

### Prerequisites

- **Python 3** is required.

### Required Python Libraries

Install the following libraries using pip:

```bash
pip install PyQt5 nltk
