import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("DB_USER"),  
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
)
engine = create_engine(DATABASE_URL)


SessionLocal = sessionmaker(
        autocommit = False,
        autoflush = False,
        bind = engine
)

class Base(DeclarativeBase):
    pass

db = SessionLocal()

# data = {
# "age": [
#     20, 25, None, 35, 40,
#     22, 28, 31, None, 45,
#     50, 27, 33, 38, 42,
#     None, 24, 29, 36, 48
# ],

# "income": [
#     30000, 40000, 50000, None, 80000,
#     35000, 45000, 52000, 60000, None,
#     90000, 42000, 55000, 65000, 75000,
#     48000, None, 47000, 62000, 85000
# ],

# "city": [
#     "Hanoi", "Hanoi", "Da Nang", "Hanoi", None,
#     "Ho Chi Minh", "Hanoi", "Da Nang", "Ho Chi Minh", "Hanoi",
#     "Da Nang", None, "Hanoi", "Ho Chi Minh", "Da Nang",
#     "Hanoi", "Ho Chi Minh", "Da Nang", "Hanoi", "Ho Chi Minh"
# ],

# "bought": [
#     0, 1, 1, 0, 1,
#     0, 1, 1, 1, 0,
#     1, 0, 1, 1, 1,
#     0, 0, 1, 0, 1
# ]
# }


# df = pd.DataFrame(data)

# df.to_sql(
#     "customers",
#     engine,
#     if_exists="append",
#     index=False
# )


