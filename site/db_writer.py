import json
from datetime import datetime
from sqlalchemy_engine import session
from models_sqlalchemy import Task

def lambda_handler(event, context):
    job_id = event['job_id']
    task_id = event['task_id']
    db_obj = event['db_obj']
    try:
        print(' working on: job_id:{}, task_id:{}'.format(job_id, task_id))
        session.query(Task).filter_by(job_id=job_id, task_id=task_id).update(db_obj)
        session.commit()
        print("Time stamp : {}".format(datetime.utcnow()))
    except Exception as e:
        session.query(Task).filter_by(job_id=job_id,task_id=task_id).update(
            task_status='error')
        raise e
    return 'Done'