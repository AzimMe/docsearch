

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
from ollama import Client

ollama = Client(host='http://localhost:11434')  

# For Ollama

import pandasai
from pandasai import SmartDataframe
from pandasai.llm.base import LLM

from pandasai.llm.base import LLM

class OllamaLLM(LLM):
    def __init__(self, model="llama3:8b",host='http://localhost:11434'):
        self.model = model
        self.client = Client(host=host)  # <-- use remote IP here

    def call(self, prompt: Any, context: dict = None) -> str:
        # Convert prompt to string if it's a Pydantic prompt object
        prompt_str = str(prompt)
        import os


        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt_str}]
        )
        return response['message']['content']

    @property
    def type(self) -> str:
        return "ollama"
# Constants
CHUNK_SIZE = 1000
TOP_K_CHUNKS_PER_CLUSTER = 5
EMBED_MODEL = 'paraphrase-MiniLM-L6-v2'
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

# ... [imports remain unchanged]

# ... [imports remain unchanged]

class DocumentProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks = []
        self.metadata = []
        self.embeddings = None
        self.query_history = []

    def process_all_files(self) -> None:
        print(f"Processing files in {self.data_path}")
        self._load_and_chunk_files()

        if not self.chunks:
            print("No valid chunks found.")
            return

        print(f"Generated {len(self.chunks)} chunks from documents")

        print("Generating embeddings...")
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

        self.embeddings = self.embedder.encode(self.chunks,batch_size=32, show_progress_bar=True)

    def _load_and_chunk_files(self) -> None:
        for filepath in glob.glob(f"{self.data_path}/**/*.*", recursive=True):
            ext = os.path.splitext(filepath)[1].lower()
            try:
                text = self._extract_text_from_file(filepath, ext)
                if not text:
                    continue

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
        if ext in ['.txt', '.md', '.csv']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if ext == '.csv':
                try:
                    df = pd.read_csv(filepath)
                    sentences = []
                    for _, row in df.iterrows():
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

    def _combine_query_with_history(self, current_query: str) -> str:
        if self.query_history:
            last_query = self.query_history[-1]
            query_embeddings = self.embedder.encode([current_query, last_query])
            similarity = cosine_similarity([query_embeddings[0]], [query_embeddings[1]])[0][0]
            decay_factor = min(similarity, 0.5)
            if decay_factor > 0.2:
                return f"{current_query}\n\nPrevious related context (for broadening only): {last_query}"
        return current_query

    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> Tuple[List[str], List[str], List[str], List[str]]:
        if self.embeddings is None or not self.chunks:
            print("No embeddings found. Please run process_all_files() first.")
            return [], [], [], []

        enhanced_query = self._enhance_query_with_date(query)
        query_embedding = self.embedder.encode([enhanced_query])[0]

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        relevant_chunks = []
        relevant_filepaths = []

        for idx in top_indices:
            relevant_chunks.append(self.chunks[idx])
            relevant_filepaths.append(self.metadata[idx]['filepath'])

        seen_files = set()
        r_list = []
        print("\nTop relevant files for your query:")
        for path in relevant_filepaths:
            if path not in seen_files:
                r_list.append(path)
                print(f"  • {os.path.basename(path)}")
                seen_files.add(path)
                if len(seen_files) >= 3:
                    break

        return relevant_chunks, [], list(seen_files), r_list

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
        relevant_chunks, _, relevant_filepaths, files_weigh = self.retrieve_relevant_chunks(query)

        if not relevant_chunks:
            return "I couldn't find any relevant information in the documents."

        def file_weight(rank: int) -> float:
            return {0: 1.0, 1: 0.5, 2: 0.25}.get(rank, 0.0)

        csv_files = [file for file in files_weigh if file.lower().endswith(".csv")]
        csv_unweight = sum(file_weight(i) for i, path in enumerate(reversed(files_weigh)) if path.lower().endswith(".csv"))
        csv_weight = csv_unweight / len(relevant_filepaths)
        print(f"\nCSV weight: {csv_weight}")

        if csv_weight > 0.4 and csv_files:
            try:
                dfs = [pd.read_csv(f) for f in csv_files]
                combined_df = pd.concat(dfs, ignore_index=True)

                sdf = SmartDataframe(combined_df, config={"llm": OllamaLLM()})
                schema_hint = ", ".join(combined_df.columns)
                prompt = (
                    f"The dataset has these columns: {schema_hint}\n"
                    f"If date operations are needed, make sure to `import datetime`.\n\n"
                    f"When asked for value return value\n\n"
                    f"prefer returning value over plot\n\n"
                    f"return plots only if explicitly asked for\n\n"
                    f"Answer alongside a few details not only a number or field\n\n"
                    f"If no result returned,no code returned or unable to answer my question is a response, retry 5 times, stop if result returned\n\n"
                    f"{query}\n\n"
                )
                return sdf.chat(prompt)
            except Exception as e:
                return f"Failed to analyze CSV files: {str(e)}"

        context = "\n\n".join(relevant_chunks)
        date_related = any(re.search(pattern, query.lower()) for pattern in DATE_PATTERNS)

        if date_related:
            prompt = (
                f"Answer the following question about time periods or dates using ONLY the information in the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"The question involves a time period or date: {query}\n\n"
                f"Focus on extracting numeric values and statistics for the specific time period. "
                f"If the exact date/period information is not in the context, say 'I don't have information for that specific time period'"
            )
        else:
            prompt = (
                f"Answer the following question using ONLY the information in the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
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


import sys
import json

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        folder_path = sys.argv[1]
        question = sys.argv[2]

        processor = DocumentProcessor(folder_path)
        processor.process_all_files()

        answer = processor.answer_query(question)
        # add this line to extract relevant file paths
        _, _, relevant_filepaths, _ = processor.retrieve_relevant_chunks(question)

        # print as JSON so the Node backend can parse it
        print(json.dumps({
            "answer": answer,
            "filepaths": relevant_filepaths
        }))
        
    else:
        print("Usage: python rag.py <folder_path> <question>")
