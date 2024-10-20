Emotion Detection using MLOps
==============================
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Features

- **Data Ingestion**: Loads tweet data with sentiment labels.
- **Preprocessing**: Includes lemmatization, stop word removal, and handling of punctuation and numbers.
- **Feature Engineering**: Utilizes the Bag of Words (BOW) model to convert text into numerical features.
- **Model Building**: Implements a Gradient Boosting Classifier for sentiment classification.
- **MLOps Integration**: 
  - **DVC**: Data Version Control for managing datasets and model versions.
  - **ML Pipeline**: Streamlined workflow from data preparation to model training and evaluation.

## Evaluation Metrics

The model's performance is assessed using:
- Accuracy
- Precision
- Recall
- ROC AUC

## Installation

To get started, clone the repository and install the required packages:

```bash
git clone https://github.com/2003HARSH/Emotion-Detection-using-MLOps
cd Emotion-Detection-using-MLOps
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Update the `params.yaml` file with relevant parameters.
2. **Run the Pipeline**: Execute the following command to run the complete workflow:

```bash
dvc repro
```

## Contributing

Contributions are welcome! Please create a new issue or submit a pull request for any enhancements or fixes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [DVC](https://dvc.org/) for data version control.
- [scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms.
