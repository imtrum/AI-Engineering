import pandas as pd
from sqlalchemy import select
import ai_engineering.models as models
from ai_engineering.data.database import db



statement = select(models.Customer)
result = db.execute(statement)
customers = result.scalars().all()

data = [
    {
        "id": customer.id,
        "age": customer.age,
        "income": customer.income,
        "city": customer.city,
    }
    for customer in customers
]

df = pd.DataFrame(data)

print(df)

# df = pd.DataFrame([
#     "age":customers.age,


# ])





