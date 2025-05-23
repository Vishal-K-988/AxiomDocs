# Axiom Docs - File Management System

A sophisticated file management system with AI capabilities, built using FastAPI and React.

## Features

- 1 Secure user authentication using Clerk
- 2 File upload and management
- 3 File renaming and organization
- 4 File deletion
- 5 Modern, responsive UI
- 6 Secure file storage using AWS S3
- 7 User-specific file organization

## Tech Stack

### Backend
- FastAPI (Python)
- SQLAlchemy (ORM)
- AWS S3 (File Storage)
- PostgreSQL (Database)

### Frontend
- React
- Tailwind CSS
- Clerk (Authentication)
- Axios (API Client)

## Prerequisites

- Python 3.8+
- Node.js 16+
- AWS Account with S3 access
- PostgreSQL database
- Clerk account for authentication

## Environment Setup

### Backend
Create a `.env` file in the `backend` directory:
```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
S3_BUCKET_NAME=your_bucket_name
DATABASE_URL=your_database_url
```

### Frontend
Create a `.env` file in the `frontend` directory:
```env
VITE_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
VITE_API_URL=http://localhost:8000
```

## Installation

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

### Files
- `POST /files/upload` - Upload a new file
- `GET /files` - List all files
- `DELETE /files/{file_id}` - Delete a file
- `PATCH /files/{file_id}/rename` - Rename a file

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── routers/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── main.py
│   ├── requirements.txt
│   └── .env
└── frontend/
    ├── src/
    │   ├── components/
    │   ├── services/
    │   └── App.jsx
    ├── package.json
    └── .env
```

## Security Features

- JWT-based authentication
- User-specific file access
- Secure file storage in S3
- Environment variable protection
- CORS configuration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Clerk for authentication
- AWS for S3 storage
- FastAPI for the backend framework
- React and Tailwind CSS for the frontend 