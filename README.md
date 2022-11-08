## How to Run the App Locally

### Using Pipenv

1. Modify the `Pipfile` to specify your version of Python 3
1. Install Flask: `pipenv install`
1. Activate the virtual environment: `pipenv shell`

### Using virtualenv and pip

1. Create the virtual environment: `virtualenv -p python3 env`
1. Activate the virtual environment: `source env/bin/activate`
1. Install Flask: `pip install Flask`

### Start Flask

Enter these two commands in your terminal:

`export FLASK_APP=app`

`flask run`