from .add_vector_store_ids import upgrade

if __name__ == "__main__":
    print("Running migration to add vector_store_ids column...")
    upgrade()
    print("Migration completed successfully!") 