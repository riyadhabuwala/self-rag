"""Document Loaders"""
import pdfplumber
from bs4 import BeautifulSoup
import requests
from typing import List, Dict

class PDFLoader:
    def load(self, file_path: str) -> List[Dict]:
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text is None:
                        continue
                        
                    raw_tables = page.extract_tables() or []
                    tables = []
                    for table in raw_tables:
                        tables.append("\n".join(" | ".join(str(c) if c is not None else "" for c in row) for row in table))
                        
                    word_count = len(text.split())
                    pages.append({
                        "page_number": i + 1,
                        "text": text,
                        "tables": tables,
                        "is_table_heavy": (len(" ".join(tables)) > 0.4 * len(text)) if text else False,
                        "word_count": word_count
                    })
        except Exception as e:
            print(f"Warning: Error loading PDF {file_path}: {e}")
            
        return pages

class HTMLLoader:
    def load(self, file_path_or_url: str) -> List[Dict]:
        html = ""
        try:
            if file_path_or_url.startswith("http"):
                response = requests.get(file_path_or_url, headers={'User-Agent': 'Sample Company sample@example.com'})
                html = response.text
            else:
                with open(file_path_or_url, "r", encoding="utf-8") as f:
                    html = f.read()
        except Exception as e:
            print(f"Error loading HTML {file_path_or_url}: {e}")
            return []
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
            
        for elem in soup.find_all(class_=lambda c: c and any(x in c for x in ["header", "footer", "nav", "sidebar", "breadcrumb"])):
            elem.decompose()
            
        main_content = soup.find(id="filing-content") or soup.find(class_="body") or soup.find("body")
        if not main_content:
            main_content = soup
            
        sections = []
        for tag in main_content.find_all(['h1', 'h2', 'h3']):
            content = []
            for nxt in tag.find_all_next():
                if nxt.name in ['h1', 'h2', 'h3']:
                    break
                content.append(nxt.get_text(separator=" ", strip=True))
                
            text = " ".join(content)
            sections.append({
                "section_title": tag.get_text(strip=True),
                "text": text,
                "page_number": None,
                "tables": [],
                "is_table_heavy": False,
                "word_count": len(text.split())
            })
            
        if not sections:
            # Fallback: Just return the whole text as one section
            text = main_content.get_text(separator="\n", strip=True)
            sections.append({
                "section_title": "Full Document",
                "text": text,
                "page_number": None,
                "tables": [],
                "is_table_heavy": False,
                "word_count": len(text.split())
            })
            
        return sections