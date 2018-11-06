
#!/bin/bash
cd ~
sudo yum -y update
sudo yum -y upgrade
sudo yum -y install blas --enablerepo=epel
sudo yum -y install lapack --enablerepo=epel
sudo yum -y install Cython --enablerepo=epel
sudo yum install python36-devel python36-pip gcc

cd kmeans-service/site
virtualenv venv --python=python3
source venv/bin/activate
pip install -r ./requirements.txt

pkgs1=~/kmeans-service/site/venv/lib64/python3.6/dist-packages/
pkgs4=~/kmeans-service/site/venv/lib64/python3.6/site-packages/
pkgs2=~/kmeans-service/site/venv/lib/python3.6/site-packages/
pkgs3=~/kmeans-service/site/venv/lib/python3.6/dist-packages/
# pkgs=pkgs

# create_job.lambda_handler
rm -f create_job.zip
zip -r9 create_job.zip create_job.py utils.py database.py
echo 'create_job.lambda_handler'

# fetch_tasks.lambda_handler
rm -f fetch_tasks.zip
zip -r9 fetch_tasks.zip fetch_tasks.py utils.py
echo 'fetch_tasks.lambda_handler'

# worker.lambda_handler
rm -f worker.zip
zip -r9 worker.zip worker.py sklearn_lite.py database.py utils.py sf_kmeans/sf_kmeans.py flask_app.py models.py config.py
current_path=$PWD
cd $pkgs1
zip -ur $current_path/worker.zip scipy/ numpy/ flask/ flask_sqlalchemy/ click/ markupsafe/ sqlalchemy/
cd $pkgs2
zip -ur $current_path/worker.zip pytz/
cd $pkgs4
zip -ur $current_path/worker.zip pandas/
cd $pkgs3
zip -ur $current_path/worker.zip werkzeug/ jinja2/ itsdangerous/ s3transfer/
cd $current_path
echo 'worker.lambda_handler'

# # report.lambda_handler
# rm -f report.zip
# zip -r9 report.zip report.py utils.py database.py
# current_path=$PWD
# cd $pkgs1
# zip -ur $current_path/report.zip pandas/ numpy/
# cd $pkgs3
# zip -ur $current_path/report.zip pytz/
# cd $current_path
# echo 'report.lambda_handler'

# # plot.lambda_handler
# rm -f plot.zip
# zip -r9 plot.zip plot.py utils.py database.py
# current_path=$PWD
# cd $pkgs1
# zip -ur $current_path/plot.zip pandas/ numpy/ matplotlib/
# cd $pkgs2
# zip -ur $current_path/plot.zip pyparsing.py cycler.py
# cd $pkgs3
# zip -ur $current_path/plot.zip pytz/ seaborn/
# cd $current_path
# echo 'plot.lambda_handler'
