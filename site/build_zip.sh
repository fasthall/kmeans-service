
#!/bin/bash

# pkgs=venv/lib/python3.6/site-packages/
pkgs=pkgs

# create_job.lambda_handler
rm -f create_job.zip
zip -r9 create_job.zip create_job.py utils.py database.py gochariots.py
current_path=$PWD
cd $pkgs
zip -ur $current_path/create_job.zip requests/ urllib3/ fnvhash/ idna/ certifi/ chardet/
cd $current_path
echo 'create_job.lambda_handler'

# fetch_tasks.lambda_handler
rm -f fetch_tasks.zip
zip -r9 fetch_tasks.zip fetch_tasks.py utils.py gochariots.py
current_path=$PWD
cd $pkgs
zip -ur $current_path/fetch_tasks.zip requests/ urllib3/ fnvhash/ idna/ certifi/ chardet/
cd $current_path
echo 'fetch_tasks.lambda_handler'

# worker.lambda_handler
rm -f worker.zip
zip -r9 worker.zip worker.py sklearn_lite.py database.py utils.py sf_kmeans/sf_kmeans.py gochariots.py
current_path=$PWD
cd $pkgs
zip -ur $current_path/worker.zip scipy/ numpy/ requests/ urllib3/ fnvhash/ idna/ certifi/ chardet/
cd $current_path
echo 'worker.lambda_handler'

# report.lambda_handler
rm -f report.zip
zip -r9 report.zip report.py utils.py database.py gochariots.py
current_path=$PWD
cd $pkgs
zip -ur $current_path/report.zip pandas/ pytz/ numpy/ requests/ urllib3/ fnvhash/ idna/ certifi/ chardet/
cd $current_path
echo 'report.lambda_handler'

# plot.lambda_handler
rm -f plot.zip
#zip -r9 plot.zip plot.py utils.py database.py gochariots.py
#current_path=$PWD
#cd $pkgs
#zip -ur $current_path/plot.zip pandas/ pytz/ numpy/ matplotlib/ scipy/ seaborn/ requests/ urllib3/ fnvhash/ idna/ certifi/ chardet/ pyparsing.py cycler.py
#cd $current_path
#echo 'plot.lambda_handler'
