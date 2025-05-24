from sqlalchemy import create_engine, Column, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

# Create engine
engine = create_engine(DATABASE_URL)

def upgrade():
    """Add vector_store_ids column to files table."""
    try:
        # Add vector_store_ids column
        with engine.connect() as conn:
            conn.execute(text("""
                ALTER TABLE files 
                ADD COLUMN vector_store_ids JSON
            """))
            conn.commit()
        print("Successfully added vector_store_ids column")
    except Exception as e:
        print(f"Error adding vector_store_ids column: {str(e)}")
        raise

def downgrade():
    """Remove vector_store_ids column from files table."""
    try:
        # SQLite doesn't support dropping columns directly
        # We need to create a new table without the column
        with engine.connect() as conn:
            # Create new table without vector_store_ids
            conn.execute(text("""
                CREATE TABLE files_new (
                    id INTEGER PRIMARY KEY,
                    filename VARCHAR,
                    original_filename VARCHAR,
                    file_type VARCHAR,
                    file_size INTEGER,
                    s3_key VARCHAR UNIQUE,
                    user_id VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    is_pdf BOOLEAN,
                    pdf_text TEXT,
                    pdf_metadata TEXT,
                    pdf_page_count INTEGER
                )
            """))
            
            # Copy data from old table to new table
            conn.execute(text("""
                INSERT INTO files_new 
                SELECT id, filename, original_filename, file_type, file_size, 
                       s3_key, user_id, created_at, updated_at, is_pdf, 
                       pdf_text, pdf_metadata, pdf_page_count 
                FROM files
            """))
            
            # Drop old table
            conn.execute(text("DROP TABLE files"))
            
            # Rename new table to old name
            conn.execute(text("ALTER TABLE files_new RENAME TO files"))
            
            conn.commit()
        print("Successfully removed vector_store_ids column")
    except Exception as e:
        print(f"Error removing vector_store_ids column: {str(e)}")
        raise

if __name__ == "__main__":
    upgrade() 