
#!/bin/bash

# create_job.lambda_handler
rm -f create_job.zip
zip -r9 create_job.zip create_job.py utils.py database.py config.py
echo 'create_job.lambda_handler'

# fetch_tasks.lambda_handler
rm -f fetch_tasks.zip
zip -r9 fetch_tasks.zip fetch_tasks.py utils.py
echo 'fetch_tasks.lambda_handler'

# worker.lambda_handler
rm -f worker.zip
zip -r9 worker.zip worker.py sklearn_lite.py database.py config.py utils.py sf_kmeans/sf_kmeans.py
cd pkgs
zip -ur ../worker.zip scipy/ numpy/
echo 'worker.lambda_handler'
