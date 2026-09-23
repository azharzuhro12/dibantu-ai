"""PostgreSQL persistence layer for DibantuAI (Step 11).

Business data (products, customers, orders, order items) lives in
PostgreSQL. The package provides:

- ``database``  engine/session management driven by ``DATABASE_URL``
- ``models``    SQLAlchemy 2.x ORM tables
- ``repository`` data access used by the business tools (no raw SQL
  scattered through the tools themselves)
- ``seed``      idempotent seed of the original mock dataset
"""

from app.db.database import dispose_engines, get_engine, session_scope
from app.db.models import Base, Customer, Order, OrderItem, Product
from app.db.seed import reset_database, seed_if_empty

__all__ = [
    "Base",
    "Customer",
    "Order",
    "OrderItem",
    "Product",
    "dispose_engines",
    "get_engine",
    "reset_database",
    "seed_if_empty",
    "session_scope",
]
