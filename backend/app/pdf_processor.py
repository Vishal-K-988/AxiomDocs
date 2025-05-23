import fitz  # PyMuPDF
import io
from typing import Optional, Dict, List

def extract_text_from_pdf(pdf_data: bytes) -> Dict[str, any]:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_data (bytes): The PDF file content as bytes
        
    Returns:
        Dict containing:
        - text: The extracted text content
        - metadata: PDF metadata
        - page_count: Number of pages
        - page_texts: List of text content per page
        
    Raises:
        ValueError: If the PDF data is invalid or empty
        Exception: For other processing errors
    """
    if not pdf_data:
        raise ValueError("PDF data is empty")
    
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        
        if not pdf_document:
            raise ValueError("Failed to open PDF document")
        
        # Extract metadata
        metadata = pdf_document.metadata or {}
        
        # Get total pages
        page_count = len(pdf_document)
        if page_count == 0:
            raise ValueError("PDF document has no pages")
        
        # Extract text from each page
        page_texts = []
        full_text = []
        
        for page_num in range(page_count):
            try:
                page = pdf_document[page_num]
                text = page.get_text()
                page_texts.append({
                    "page_number": page_num + 1,
                    "text": text
                })
                full_text.append(text)
            except Exception as page_error:
                print(f"Error processing page {page_num + 1}: {str(page_error)}")
                # Continue with other pages even if one fails
        
        # Close the document
        pdf_document.close()
        
        if not full_text:
            raise ValueError("No text could be extracted from the PDF")
        
        return {
            "text": "\n".join(full_text),
            "metadata": metadata,
            "page_count": page_count,
            "page_texts": page_texts
        }
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def is_pdf_file(filename: str) -> bool:
    """
    Check if a file is a PDF based on its extension.
    
    Args:
        filename (str): The name of the file
        
    Returns:
        bool: True if the file is a PDF, False otherwise
    """
    return filename.lower().endswith('.pdf') 