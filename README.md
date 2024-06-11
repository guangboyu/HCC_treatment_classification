## Introduction
Code implementatation for paper: [MRI radiomics to monitor therapeutic outcome of sorafenib plus IHA transcatheter NK cell combination therapy in hepatocellular carcinoma](https://link.springer.com/article/10.1186/s12967-024-04873-w)


Build machine learning models to predict treatment outcomes:
* treatment group vs. untreatment (control) group
* treatment outcomes
    * Control
    * NK
    * Sorafenib
    * Combination

## Project Structure
`mri_preprocess.py` -> `feature_extraction.py` -> `modeling.py`

`data_preprocess.py`: preprocess the data

`feature_extraction.py`: feature extraction and selection

`modeling.py`: machine learning model training, evaluation (new)

`Params.yaml`: parameters for PyRadiomics features

## Progress
