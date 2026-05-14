import re

class FinancialExtractor:
    def __init__(self):
        # Precompile regex patterns
        # tickers
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        # amounts
        self.amount_pattern_1 = re.compile(r'(?:\\?\$|\$)\\?[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K|T)?', re.IGNORECASE)
        self.amount_pattern_2 = re.compile(r'USD\s*[\d,]+(?:\.\d+)?', re.IGNORECASE)
        self.amount_pattern_3 = re.compile(r'(?:\\?\$|\$)\\?\d+(?:\.\d+)?(?:B|M|K|T| billion| million)?', re.IGNORECASE)
        # dates
        self.date_pattern = re.compile(r'\b(?:Q[1-4][\s-]?20\d{2}|FY\s?20\d{2}|fiscal year\s+20\d{2}|H[12]\s+20\d{2})\b', re.IGNORECASE)
        
    def extract_metadata_from_text(self, text: str) -> dict:
        sample = text[:5000]
        
        company_name = None
        for pattern in [r"UNITED STATES SECURITIES AND EXCHANGE COMMISSION", r"FORM 10-K", r"FORM 10-Q"]:
            match = re.search(pattern + r"\s*\n+([^\n]+)", sample, re.IGNORECASE)
            if match and match.group(1).strip() and not match.group(1).strip().isupper():
                company_name = match.group(1).strip()
                break
        
        if not company_name:
            match = re.search(r"(?:Company|Registrant):\s*([a-zA-Z0-9\s.,&]+)", sample, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip()
        
        ticker = None
        ticker_match = re.search(r"(?:NYSE|NASDAQ|Ticker Symbol):\s*([A-Z]{1,5})", sample, re.IGNORECASE)
        if ticker_match:
            ticker = ticker_match.group(1).upper()
        else:
            ticker_match = re.search(r"\(([A-Z]{1,5})\)", sample)
            # simplistic fallback check
            if ticker_match:
                ticker = ticker_match.group(1)

        fiscal_year = None
        fy_match = re.search(r"fiscal year ended [a-zA-Z]+ \d{1,2}, (\d{4})", sample, re.IGNORECASE)
        if not fy_match:
            fy_match = re.search(r"for the year ended [a-zA-Z]+ \d{1,2}, (\d{4})", sample, re.IGNORECASE)
        if not fy_match:
            fy_match = re.search(r"Annual Report (\d{4})", sample, re.IGNORECASE)
        
        if fy_match:
            fiscal_year = f"FY{fy_match.group(1)}"

        filing_date = None
        fd_match = re.search(r"(?:Date of Report|Filed|Date filed):\s*([A-Za-z]+ \d{1,2}, \d{4})", sample, re.IGNORECASE)
        if fd_match:
            filing_date = fd_match.group(1)
            
        doc_type = None
        doc_sample = sample[:2000].lower()
        if "form 10-k" in doc_sample:
            doc_type = "10-K"
        elif "form 10-q" in doc_sample:
            doc_type = "10-Q"
        elif "earnings call transcript" in doc_sample:
            doc_type = "earnings-transcript"
        elif "annual report" in doc_sample:
            doc_type = "annual-report"

        return {
            "company_name": company_name,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "filing_date": filing_date,
            "doc_type": doc_type
        }

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
        metrics_list = ["total revenue", "net revenue", "net income", "gross profit", "operating income", "EBITDA", "earnings per share", "EPS", "free cash flow", "operating cash flow", "gross margin", "operating margin", "net margin", "return on equity", "ROE", "return on assets", "ROA", "debt to equity", "current ratio", "Tier 1 capital", "Tier 1 capital ratio", "Basel III", "leverage ratio", "liquidity coverage ratio", "CET1"]
        regulatory_list = ["SEC", "GAAP", "IFRS", "PCAOB", "FASB", "10-K", "10-Q", "8-K", "proxy statement", "material weakness", "going concern", "restatement", "insider trading", "Sarbanes-Oxley", "SOX", "Dodd-Frank", "FINRA", "CFTC", "OCC", "Federal Reserve", "stress test", "DFAST", "CCAR"]

        # find tickers
        raw_tickers = self.ticker_pattern.findall(text)
        tickers = []
        for match in re.finditer(r'\b[A-Z]{1,5}\b', text):
            t = match.group(0)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            if re.search(r'\$|NYSE:|NASDAQ:|\(|stock|shares', context, re.IGNORECASE):
                tickers.append(t)
        
        amounts = self.amount_pattern_1.findall(text) + self.amount_pattern_2.findall(text) + self.amount_pattern_3.findall(text)
        amounts = [re.sub(r',|\\', '', a).strip() for a in amounts]

        dates = self.date_pattern.findall(text)
        dates = [re.sub(r'\s+', ' ', d).upper().strip() for d in dates]

        metrics = []
        text_lower = text.lower()
        for m in metrics_list:
            if m.lower() in text_lower:
                metrics.append(m)

        regulatory_terms = []
        for r in regulatory_list:
            if r.lower() in text_lower:
                regulatory_terms.append(r)

        return {
            "tickers": list(set(tickers)),
            "amounts": list(set(amounts)),
            "dates": list(set(dates)),
            "metrics": list(set(metrics)),
            "regulatory_terms": list(set(regulatory_terms))
        }

    def classify_query_intent(self, query: str) -> dict:
        query_lower = query.lower()
        
        advice_kw = ["should i buy", "should i sell", "is it a good investment", "recommend", "will the stock", "price target", "buy or sell", "worth investing", "predict", "forecast stock"]
        factual_kw = ["what was", "how much", "what were", "total", "revenue", "income", "ratio", "reported"]
        analytical_kw = ["why did", "how did", "compare", "trend", "changed", "improved", "declined", "analysis", "explain"]
        comparison_kw = ["vs", "versus", "compared to", "better than", "relative to", "peer", "industry average"]

        intent = "general"
        requires_disclaimer = False
        
        if any(k in query_lower for k in comparison_kw):
            intent = "comparison"
        elif any(k in query_lower for k in analytical_kw):
            intent = "analytical"
        elif any(k in query_lower for k in factual_kw):
            intent = "factual"
            
        if any(k in query_lower for k in advice_kw):
            intent = "advice"
            requires_disclaimer = True

        ext = self.extract_financial_entities(query)
        
        return {
            "intent": intent,
            "requires_disclaimer": requires_disclaimer,
            "disclaimer_reason": "Advice" if requires_disclaimer else None,
            "extracted_tickers": ext["tickers"],
            "extracted_fiscal_periods": ext["dates"]
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
