import pdfplumber
from PyPDF2 import PdfReader
from typing import Optional
import io

class PDFProcessor:
    def __init__(self):
        self.current_file = None
        
    def extract_text_from_pdf(self, pdf_file: bytes) -> Optional[str]:
        """Extract text from a PDF file using both pdfplumber and PyPDF2 for better results."""
        try:
            # Create a file-like object from the bytes
            pdf_stream = io.BytesIO(pdf_file)
            
            # Try pdfplumber first
            with pdfplumber.open(pdf_stream) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    
            # If pdfplumber didn't extract much text, try PyPDF2
            if len(text.strip()) < 100:  # Arbitrary threshold
                pdf_stream.seek(0)
                reader = PdfReader(pdf_stream)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                    
            return text.strip()
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None
            
    def process_pdf(self, pdf_file: bytes) -> Optional[str]:
        """Process a PDF file and return its text content."""
        self.current_file = pdf_file
        return self.extract_text_from_pdf(pdf_file) 