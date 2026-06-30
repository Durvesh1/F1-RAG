from database_service.db_connection import engine

from models import *
from models.base import Base


Base.metadata.create_all(engine)