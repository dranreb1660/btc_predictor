# BTC_PREDICTOR

Predict clossing price of bitcoin using historic data

## Project Structure

    ├── README.md                       <- Project information, and steps to reproduce the Project
    ├── data
    │   ├── external                    <- Data from third party sources.
    │   ├── interim                     <- Intermediate data that has been transformed.
    │   ├── processed                   <- The final, canonical data sets for modeling.
    │   └── raw                         <- The original, immutable data dump from url.
    │
    ├── logs                            <- Log run info. exampl. checkpoints.
    │
    ├── docs                            <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                          <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                   the creator's initials, and a short `-` delimited description, e.g.
    │                                   `1.0-sa-initial-data-exploration`.
    ├──Pipfile & Pipfile.lock           <- acts a recquirements.txt for pipenv environment
    |
    ├── references                      <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                     <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.generated with `pip freeze > requirements.txt`
    │
    ├──script.sh                        <- This will be the file to set the project environment
    │

    ├── btc_predictor
    │       ├── __init__.py             <- Makes src a Python module
    │       │
    │       ├── config.py               <- Configuration file for some global variables
    │       │
    │       ├── data                    <- Scripts to download or generate data
    |       |   |
    |       |   ├── base_dataset.py     <- creates a pytorch dataset from the data
    |       |   ├── lit_data_module.py  <- wraps the dataset from the litng predictor
    │       │   └── prep_n_build_f.py   <- script to download tan prepare raw data
    │       │
    │       ├── models                  <- Scripts to model architecture and data modules
    │       │   │
    │       │   ├── base_model.py
    │       │   └── lit_model.py
    │       │
    |       ├── predict.py              <- script to make predictions on a test set
    |       ├── train.py                <- script to train the model
    |       |
    │       └── visualization           <- Scripts to create exploratory and results oriented visualizations
    │           └── visualize.py
    │
    ├── tests                           <- Test scripts for unit testing (e.g. using pytest),
    │                                   performance and load testing of the API
    │
    ├── api.py                          <- Flask API script
    │
    └── .gitignore

### Installing development requirements

---

    bash script.sh. This is the first file the run. it sets the python path for terminal session and sets pipenv with with the required dependencies.

    add export PYTHONPATH=.:$PYTHONPATH to your ~/.bashrc and source ~/.bashrc for a more permanat approach to setting the python path.

    pip install -r requirements.txt

### Running the tests

---

    pytest tests

### Build documentation using Sphinx

---

    cd docs/
    make html

---

<p><small>Project created using the <a target="_blank" href="https://github.com/sujitahirrao/flask-ai-api-template">Flask AI API Template</a>.</small></p>
