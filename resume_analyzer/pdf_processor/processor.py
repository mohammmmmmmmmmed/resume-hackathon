from typing import Dict, List, Optional, Any
import pdfplumber
from pathlib import Path
import logging
import io
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and text extraction from resumes."""
    
    def __init__(self):
        self.supported_extensions = {'.pdf'}
        self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file is a supported PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
            
        if path.suffix.lower() not in self.supported_extensions:
            logger.error(f"Unsupported file type: {path.suffix}")
            return False
            
        return True
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text if successful, None otherwise
        """
        if not self.validate_file(file_path):
            return None
            
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def extract_tables(self, file_path: str) -> List[Dict]:
        """
        Extract tables from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[Dict]: List of extracted tables
        """
        if not self.validate_file(file_path):
            return []
            
        try:
            with pdfplumber.open(file_path) as pdf:
                tables = []
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                return tables
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {str(e)}")
            return []
    
    def process_resume(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process resume PDF and extract structured information."""
        try:
            # Extract text from PDF
            text = self.process_pdf(pdf_content)
            
            if not text:
                logger.error("No text extracted from PDF")
                return {}
            
            # Log the extracted text for debugging
            logger.info("Extracted text from PDF:")
            logger.info(text)
            
            return {"text": text}
            
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            return {}

    def process_pdf(self, pdf_content: bytes) -> str:
        """Process PDF content and extract text."""
        try:
            # Create a temporary file-like object from the PDF content
            pdf_file = io.BytesIO(pdf_content)
            
            # Extract text from PDF
            text_content = []
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
                        logger.info(f"Extracted text from page:")
                        logger.info(text)
            
            # Join all pages with newlines
            return '\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return ""
