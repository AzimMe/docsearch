import os
import glob
import re
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set

# For embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# For file processing
import zipfile
import csv
from io import StringIO

# For PDF support (you'll need to install pypdf)
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PyPDF not installed. PDF support disabled.")

# For XLSX support (you'll need to install openpyxl)
try:
    import openpyxl
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False
    print("Openpyxl not installed. Excel support disabled.")

# For Ollama
import ollama

# Constants
CHUNK_SIZE = 1000
TOP_K_CHUNKS_PER_CLUSTER = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3:8b"
CATEGORIES = ["invoice", "financial", "student data", "salary", "report", "sales", "other"]

# Date pattern for improved date handling
DATE_PATTERNS = [
    r'(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+\d{4}',
    r'\d{1,2}/\d{1,2}/\d{2,4}',
    r'\d{4}-\d{1,2}-\d{1,2}',
    r'q[1-4]\s+\d{4}',
    r'quarter\s+[1-4]\s+\d{4}',
    r'fy\s*\d{2,4}'
]

class DocumentProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks = []
        self.metadata = []
        self.tags = []
        self.embeddings = None
        self.file_categories = {}
        self.query_history=[]
        
    def process_all_files(self) -> None:
        """Process all files in the data path"""
        print(f"Processing files in {self.data_path}")
        self._load_and_chunk_files()
        
        if not self.chunks:
            print("No valid chunks found.")
            return
            
        print(f"Generated {len(self.chunks)} chunks from documents")
        
        # Extract tags for each chunk
        self.tags = self._get_chunk_tags(self.chunks)
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.embedder.encode(self.chunks, show_progress_bar=True)
        
        # Categorize files
        self._categorize_files()
        
    def _load_and_chunk_files(self) -> None:
        """Load all files and split them into chunks"""
        for filepath in glob.glob(f"{self.data_path}/**/*.*", recursive=True):
            ext = os.path.splitext(filepath)[1].lower()
            
            try:
                text = self._extract_text_from_file(filepath, ext)
                if not text:
                    continue
                    
                # Chunk the text
                for i in range(0, len(text), CHUNK_SIZE):
                    chunk = text[i:i+CHUNK_SIZE]
                    if chunk.strip():
                        self.chunks.append(chunk)
                        self.metadata.append({
                            'filepath': filepath, 
                            'chunk_id': i // CHUNK_SIZE,
                            'filename': os.path.basename(filepath)
                        })
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    def _extract_text_from_file(self, filepath: str, ext: str) -> str:
        """Extract text from a file based on its extension"""
        if ext in ['.txt', '.md', '.csv']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
            # For CSV, convert to formatted text
            if ext == '.csv':
                try:
                    df = pd.read_csv(filepath)

                    sentences = []
                    for _, row in df.iterrows():
                        # Format: Field1: value1, Field2: value2, ...
                        sentence = ', '.join(f"{col}: {row[col]}" for col in df.columns)
                        sentences.append(sentence + '.')
                    return ' '.join(sentences)

                except:
                    pass
            return text
            
        elif ext == '.pdf' and PDF_SUPPORT:
            text = ""
            with open(filepath, 'rb') as f:
                pdf = PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
                print(text)
            return text
            
        elif ext in ['.xlsx', '.xls'] and EXCEL_SUPPORT:
            text = f"Excel File: {os.path.basename(filepath)}\n\n"
            df = pd.read_excel(filepath)
            text += df.to_string(index=False)
            return text
            
        elif ext == '.zip':
            text = ""
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith(('.txt', '.csv')):
                        with zip_ref.open(file) as f:
                            text += f.read().decode('utf-8', errors='ignore') + "\n\n"
            return text
            
        return ""
        
    def _get_chunk_tags(self, chunks: List[str], n: int = 5) -> List[List[str]]:
        """Extract top n tags per chunk using TF-IDF"""
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(chunks)
        tags_per_chunk = []
        for i in range(X.shape[0]):
            row = X[i].toarray().flatten()
            top_indices = row.argsort()[::-1][:n]
            tags = [vectorizer.get_feature_names_out()[j] for j in top_indices]
            tags_per_chunk.append(tags)
        return tags_per_chunk
    
    def _combine_query_with_history(self, current_query: str) -> str:
        if self.query_history:
            last_query = self.query_history[-1]
            query_embeddings = self.embedder.encode([current_query, last_query])
            similarity = cosine_similarity([query_embeddings[0]], [query_embeddings[1]])[0][0]
            decay_factor = min(similarity, 0.5)
            if decay_factor > 0.2:
                return f"{current_query}\n\nPrevious related context (for broadening only): {last_query}"
        return current_query
    
    def _categorize_files(self) -> None:
        """Categorize each file based on its content"""
        file_chunks = defaultdict(list)
        file_metadata = defaultdict(list)
        
        for i, meta in enumerate(self.metadata):
            filepath = meta['filepath']
            file_chunks[filepath].append(self.chunks[i])
            file_metadata[filepath].append(meta)
        
        print("\n--- File Classification Results ---")
        
        for filepath, chunks in file_chunks.items():
            keywords = self._get_top_keywords(chunks, n=7)
            sample_text = chunks[0]
            category = self._classify_file(keywords, sample_text)
            self.file_categories[filepath] = category
            
            print(f"\nCategory: {category}")
            print(f"  📄 {os.path.basename(filepath)}")
            print(f"  Keywords: {', '.join(keywords)}")
    
    def _get_top_keywords(self, texts: List[str], n: int = 5) -> List[str]:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        indices = X.sum(axis=0).A1.argsort()[::-1][:n]
        return [vectorizer.get_feature_names_out()[i] for i in indices]
    
    def _classify_file(self, keywords: List[str], sample_text: str) -> str:
        prompt = (
            f"Given the following keywords from a document: {', '.join(keywords)}.\n"
            f"And a sample of the content:\n{sample_text[:500]}...\n\n"
            f"Classify this document into one of the following categories: {', '.join(CATEGORIES)}.\n"
            f"Only return the category name - no additional text or explanation."
        )
        
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        category = response['message']['content'].strip().lower()
        
        for c in CATEGORIES:
            if c.lower() in category:
                return c
        
        return "other"
        
    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> Tuple[List[str], List[str], List[str]]:
        if self.embeddings is None or not self.chunks:
            print("No embeddings found. Please run process_all_files() first.")
            return [], [], []

        enhanced_query = self._enhance_query_with_date(query)
        query_embedding = self.embedder.encode([enhanced_query])[0]

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        relevant_chunks = []
        relevant_tags = []
        relevant_filepaths = []

        for idx in top_indices:
            relevant_chunks.append(self.chunks[idx])
            relevant_tags.append(", ".join(self.tags[idx]))
            relevant_filepaths.append(self.metadata[idx]['filepath'])

        seen_files = set()
        print("\nTop relevant files for your query:")
        for path in relevant_filepaths:
            if path not in seen_files:
                category = self.file_categories.get(path, "unknown")
                print(f"  • {os.path.basename(path)} [{category}]")
                seen_files.add(path)
                if len(seen_files) >= 3:
                    break

        return relevant_chunks, relevant_tags, list(seen_files)

    def _enhance_query_with_date(self, query: str) -> str:
        query_lower = query.lower()
        date_matches = []
        for pattern in DATE_PATTERNS:
            matches = re.findall(pattern, query_lower)
            if matches:
                date_matches.extend(matches)
        
        if date_matches:
            enhanced = query + " " + " ".join([
                "time period", "date range", "temporal data", 
                "chronological information", "time series"
            ])
            return enhanced
            
        return query
        
    def answer_query(self, query: str) -> str:
        self.query_history.append(query)
        relevant_chunks, relevant_tags, relevant_files = self.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information in the documents."
        
        context = "\n\n".join(relevant_chunks)
        tag_context = "\n".join(relevant_tags)
        hist=self._combine_query_with_history(query)
        date_related = any(re.search(pattern, query.lower()) for pattern in DATE_PATTERNS)
        
        if date_related:
            prompt = (
                f"Answer the following question about time periods or dates using ONLY the information in the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"Tags: {tag_context}\n\n"
                f"The question involves a time period or date: {query}\n\n"
                f"The history of previous chat - please use when necessary:{hist}\n\n"
                f"Focus on extracting numeric values and statistics for the specific time period. "
                f"If the exact date/period information is not in the context, say 'I don't have information for that specific time period.'"
            )
        else:
            prompt = (
                f"Answer the following question using ONLY the information in the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"Tags: {tag_context}\n\n"
                f"Question: {query}\n\n"
                f"The history of previous chat - please use when necessary:{hist}\n\n"
                f"Give a concise, accurate answer based only on the provided information. "
                f"If the answer isn't in the context, say 'I don't have that information.'"
            )
        
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response['message']['content']


def main():
    data_path = input("Enter the folder path: ")
    processor = DocumentProcessor(data_path)
    processor.process_all_files()
    
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        answer = processor.answer_query(query)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
