# dtu_mlops_age_prediction

Age prediction model for facial images

## Project structure

The directory structure of the project looks like this:

```txt 
├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│   └── README.md        <- Report of the project
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│   ├── test_data.py     <- Test data integrity
│   │
│   └── test_model.py    <- Test model architecture and output
│
├── config  <- Configuration files directory
│   │
│   ├── config_train.yaml      <- Configuration redirecting to experiments
│   │ 
│   ├── wandb_sweep.yaml      <- Configuration with the sweep parameters
│   │
│   └── experiment             <- Experiment folder
│       ├── exp1.yaml
│       └── exp2.yaml
│
├── dockerfiles  <- Folder with dockerfiles for creating images of each script execution
│   
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── api    <- Scripts for deployment
│   │   |
│   │   └── main.py      <- Script for opening the local model deployment api
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   │
│   ├── train_model.py   <- script for training the model
│   │
│   └── train_model_exp.py   <- script for training the model from the experiments configuration
│   │
│   └── train_model_sweep.py   <- script for training the model executing a sweep with wandb
│   │
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
