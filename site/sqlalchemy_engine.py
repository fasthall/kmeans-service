from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import POSTGRES_URI
from models_sqlalchemy import Base, Job, Task

engine = create_engine(POSTGRES_URI)
Base.metadata.bind = engine
 
DBSession = sessionmaker(bind=engine)
session = DBSession()