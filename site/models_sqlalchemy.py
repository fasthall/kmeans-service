# These are raw data models for native sqlalchemy engine (not flask-sqlalchemy)

import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, ARRAY, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
 
Base = declarative_base()
 
class Job(Base):
    __tablename__ = 'job'
    job_id = Column(Integer, primary_key=True)
    n_experiments = Column(Integer)
    max_k = Column(Integer)
    n_init = Column(Integer)
    n_tasks =  Column(Integer)
    columns = Column(ARRAY(String))
    filename = Column(String(100))
    start_time = Column(DateTime)
    scale = Column(Boolean)
    s3_file_key = Column(String(200))
    
 
class Task(Base):
    __tablename__ = 'task'
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer)
    job_id = Column(Integer)
    n_init =  Column(Integer)
    n_tasks =  Column(Integer)
    n_experiments =  Column(Integer)
    max_k =  Column(Integer)
    k =  Column(Integer)
    covar_type = Column(String(10))
    covar_tied = Column(Boolean)
    task_status = Column(String(10))
    columns = Column(ARRAY(String))
    filename = Column(String(100))
    s3_file_key = Column(String(200))
    start_time = Column(DateTime)
    scale = Column(Boolean)
    aic = Column(Float)
    bic = Column(Float)
    labels = Column(ARRAY(Integer))
    iteration_num = Column(Integer)
    centers = Column(ARRAY(Float))
    cluster_counts = Column(ARRAY(Integer))
    cluster_count_minimum = Column(Integer)
    elapsed_time = Column(Integer)
    elapsed_read_time = Column(Integer)
    elapsed_processing_time = Column(Integer)