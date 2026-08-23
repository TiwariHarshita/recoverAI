from app.db.database import create_schema, get_database_url


def main() -> None:
    create_schema()
    print(f"RecoverAI database schema ready: {get_database_url()}")


if __name__ == "__main__":
    main()
