## Introduction
Build machine learning models to predict treatment outcomes:
* treatment group vs. untreatment (control) group
* treatment outcomes
    * Control
    * NK
    * Sorafenib
    * Combination

## Project Structure
`data_preprocess.py` -> `feature_extraction.py` -> `model.py`

`data_preprocess.py`: preprocess the data

`feature_extraction.py`: feature extraction and selection

`model.py`: machine learning model training, evaluation

`Params.yaml`: parameters for PyRadiomics features