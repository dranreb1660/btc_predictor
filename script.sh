

apt-get update
apt-get install vim
pip install pipenv
export PYTHONPATH=.:$PYTHONPATH
echo 'Now export PYTHONPATH=.:$PYTHONPATH before source ~/.bashrc'
pipenv install
pipenv shell



