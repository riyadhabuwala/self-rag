"""Adaptive Chunking Module"""
import re
from typing import List, Dict

class ChunkingStrategy:
    def chunk_prose(self, text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            
            if current_word_count + word_count > chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                if len(chunk_str.split()) >= 30:
                    chunks.append(chunk_str)
                
                # Overlap logic
                overlap_words = []
                overlap_count = 0
                for s in reversed(current_chunk):
                    s_words = s.split()
                    if overlap_count + len(s_words) <= chunk_overlap:
                        overlap_words.insert(0, s)
                        overlap_count += len(s_words)
                    else:
                        break
                        
                current_chunk = overlap_words + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_word_count += word_count
                
        if current_chunk:
            chunk_str = " ".join(current_chunk)
            if len(chunk_str.split()) >= 30:
                chunks.append(chunk_str)
                
        return chunks

    def chunk_table(self, table_text: str) -> List[str]:
        words = table_text.split()
        if len(words) <= 800:
            return ["TABLE: " + table_text]
            
        lines = table_text.split('\n')
        if not lines:
            return []
            
        header = lines[0]
        chunks = []
        current_chunk = [header]
        current_word_count = len(header.split())
        
        for line in lines[1:]:
            line_word_count = len(line.split())
            if current_word_count + line_word_count > 800:
                chunks.append("TABLE: " + "\n".join(current_chunk))
                current_chunk = [header, line]
                current_word_count = len(header.split()) + line_word_count
            else:
                current_chunk.append(line)
                current_word_count += line_word_count
                
        if len(current_chunk) > 1:
            chunks.append("TABLE: " + "\n".join(current_chunk))
            
        return chunks

    def chunk_hierarchical(self, text: str, doc_title: str) -> dict:
        words = text.split()
        summary = " ".join(words[:200]) # Fallback summary
        return {
            "summary": summary,
            "detail_chunks": self.chunk_prose(text)
        }

    def detect_chunk_type(self, text: str) -> str:
        lines = text.strip().split('\n')
        
        if "|" in text or any(re.match(r'^\s*\d', line) for line in lines):
            if "TABLE:" in text:
                return "table"
            
        if all(line.startswith(' ') or line.startswith('\t') for line in lines if line.strip()):
            return "code"
            
        if len(text.split()) < 15 and not re.search(r'[.?!]$', text.strip()):
            return "header"
            
        return "prose"