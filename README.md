# File Management System with AI Capabilities

## Project Overview
This project is a sophisticated file management system that combines modern web technologies with AI capabilities. It features a React-based frontend and a FastAPI backend, providing a seamless experience for file management, PDF processing, and AI-powered document interaction.

## Architecture

### Frontend
- Built with React and TypeScript
- Uses Tailwind CSS for styling
- Features a modern, responsive UI with a collapsible sidebar
- Implements real-time chat interface for document interaction

### Backend
- FastAPI-based REST API
- SQLAlchemy for database management
- PDF processing capabilities using PyMuPDF
- Vector store integration for AI features
- CORS middleware for secure cross-origin requests

## Key Features

### 1. File Management
```python
# Backend file handling
@app.post("/api/files/upload")
async def upload_file(file: UploadFile):
    # File processing logic
    return {"filename": file.filename, "status": "success"}
```

### 2. PDF Processing
```python
def extract_text_from_pdf(pdf_data: bytes) -> Dict[str, any]:
    """
    Extract text content from a PDF file with metadata and page information
    """
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    metadata = pdf_document.metadata
    page_count = len(pdf_document)
    # ... text extraction logic
```

### 3. Chat Interface
The system includes an AI-powered chat interface that allows users to interact with their documents:
```jsx
<ChatInterface
  selectedFile={selectedFile}
  chatHistory={chatHistory}
  onSendMessage={handleSendMessage}
  isLoading={isChatLoading}
  userProfileImage={userProfileImage}
/>
```

## Development Challenges and Solutions

### 1. PDF Processing
**Challenge**: Efficiently extracting and processing large PDF files while maintaining performance.
**Solution**: Implemented streaming processing and caching mechanisms:
```python
# PDF processing with error handling
try:
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    # Process pages individually
    for page_num in range(page_count):
        page = pdf_document[page_num]
        text = page.get_text()
        # Store in vector database
except Exception as e:
    handle_error(e)
```

### 2. Real-time Chat
**Challenge**: Managing real-time communication between frontend and backend.
**Solution**: Implemented WebSocket connections and state management:
```javascript
const handleSendMessage = async (message) => {
  try {
    setIsLoading(true);
    const response = await sendMessageToBackend(message);
    updateChatHistory(response);
  } catch (error) {
    handleError(error);
  } finally {
    setIsLoading(false);
  }
};
```

### 3. File Management
**Challenge**: Handling large file uploads and maintaining file integrity.
**Solution**: Implemented chunked uploads and validation:
```python
@app.post("/api/files/upload")
async def upload_file(file: UploadFile):
    try:
        # Validate file type and size
        if not is_valid_file(file):
            raise HTTPException(status_code=400, detail="Invalid file")
        
        # Process file in chunks
        contents = await file.read()
        # Store in database
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Error Handling

The system implements comprehensive error handling at multiple levels:

1. **Frontend Error Handling**:
```jsx
{error && (
  <div className="bg-red-50 border-l-4 border-red-400 p-4">
    <div className="flex">
      <div className="ml-3">
        <p className="text-sm text-red-700">{error}</p>
      </div>
    </div>
  </div>
)}
```

2. **Backend Error Handling**:
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
```

## Setup and Installation

1. **Backend Setup**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. **Frontend Setup**:
```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

Create a `.env` file in the backend directory:
```
DATABASE_URL=sqlite:///./file_management.db
VECTOR_STORE_PATH=./vector_store
```

## API Documentation

The API documentation is available at `/docs` when running the backend server. Key endpoints include:

- `POST /api/files/upload` - Upload new files
- `GET /api/files` - List all files
- `POST /api/chat` - Send chat messages
- `GET /api/vector-store/search` - Search through document content

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 