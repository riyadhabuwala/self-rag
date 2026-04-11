"""Financial Entity and Metadata Extractor"""
import re

class FinancialExtractor:
    def extract_metadata_from_filename(self, filename: str) -> dict:
        ticker = re.search(r'([A-Z]{1,5})', filename)
        doc_type = re.search(r'(10-K|10-Q|earnings-transcript|annual-report)', filename, re.IGNORECASE)
        fiscal_year = re.search(r'(FY20\d{2}|Q[1-4]-20\d{2})', filename, re.IGNORECASE)
        filing_date = re.search(r'\d{4}-\d{2}-\d{2}', filename)
        
        return {
            "ticker": ticker.group(1) if ticker else None,
            "doc_type": doc_type.group(1).upper() if doc_type else None,
            "fiscal_year": fiscal_year.group(1).upper() if fiscal_year else None,
            "filing_date": filing_date.group(0) if filing_date else None
        }

    def extract_financial_entities(self, text: str) -> dict:
        return {
            "tickers": list(set(re.findall(r'\$[A-Z]{1,5}\b', text))),
            "amounts": list(set(re.findall(r'\$\d+(?:\.\d+)?(?:B|M| billion| million|,\d{3})?', text))),
            "dates": list(set(re.findall(r'\b(?:Q[1-4] 20\d{2}|FY20\d{2}|fiscal year 20\d{2}|H[12] 20\d{2})\b', text, re.IGNORECASE))),
            "metrics": [m for m in ["revenue", "net income", "EPS", "EBITDA", "gross margin", "operating income", "free cash flow", "ROE", "ROA", "P/E ratio", "market cap", "dividend yield", "Tier 1 Capital", "Basel III", "leverage ratio", "liquidity ratio"] if m.lower() in text.lower()],
            "terms": [t for t in ["SEC", "GAAP", "IFRS", "10-K", "10-Q", "8-K", "proxy statement", "material weakness", "going concern", "restatement", "insider trading"] if t.lower() in text.lower()]
        }

    def clean_text(self, text: str) -> str:
        text = re.sub(r' {3,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.replace('\x00', '')
        
        lines = text.split('\n')
        cleaned = []
        for l in lines:
            if not re.match(r'^(Page \d+ of \d+|EDGAR Filing|.*Copyright.*|https?://\S+)$', l.strip(), re.IGNORECASE):
                cleaned.append(l)
                
        return '\n'.join(cleaned)