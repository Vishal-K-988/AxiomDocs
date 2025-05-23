import os
from app.db.vector_store import vector_store
from app.utils.embeddings import prepare_documents_for_vector_store
import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def test_pdf_upload(pdf_path: str):
    """Test PDF upload and vector storage."""
    try:
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from PDF")
        
        # Prepare metadata
        metadata = {
            "source": os.path.basename(pdf_path),
            "type": "pdf",
            "file_path": pdf_path
        }
        
        # Prepare documents for vector store
        chunks, embeddings, chunk_metadata = prepare_documents_for_vector_store(
            text=text,
            metadata=metadata
        )
        
        print(f"Created {len(chunks)} chunks with embeddings")
        
        # Add to vector store
        doc_ids = vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadata=chunk_metadata
        )
        
        print(f"Successfully added documents to vector store with IDs: {doc_ids}")
        
        # Test search functionality
        test_query = chunks[0][:100]  # Use first 100 chars of first chunk as test query
        print("\nTesting search with query:", test_query)
        
        from app.utils.embeddings import generate_embedding
        query_embedding = generate_embedding(test_query)
        results = vector_store.search_similar(query_embedding, n_results=3)
        
        print("\nSearch results:")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        )):
            print(f"\nResult {i+1}:")
            print(f"Distance: {dist}")
            print(f"Source: {meta['source']}")
            print(f"Chunk {meta['chunk_index'] + 1} of {meta['total_chunks']}")
            print(f"Preview: {doc[:200]}...")
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        print("\nCollection statistics:")
        print(f"Total documents: {stats['count']}")
        print(f"Total vectors: {stats['total_vectors']}")
        print(f"Dimension: {stats['dimension']}")
        
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Replace this with your PDF file path
    pdf_path = "path/to/your/test.pdf"
    test_pdf_upload(pdf_path) 