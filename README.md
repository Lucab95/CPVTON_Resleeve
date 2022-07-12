CPVTON resleeve assesment
==============================

Assesment for Resleeve, inference and test new images

Project Organization
------------

    ├── LICENSE
    ├── 
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── test           <- test set coming from the CPVTON dataset.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- Initial images as they are.
    │
    ├── checkpoints        <- Checkpoints for GMM,TOM and graphonomy.
    ├── graphonomy         <- graphonomy directory to perform human parsing
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── openpose_api       <- python_api for openpose
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    ├── result            <- Results of the assesment.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to handle or generate data
    │   |   |── cp_dataset 
    │   │   └── build_image_mask.py <- Script to build the image and cloth mask.
    │   ├               
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── networks.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── test_extdata.py    <- Script to preprocess and predict the external data.
    |
    └── test_pretrained.py <- Script to inference the test data.



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
"# CPVTON_Resleeve" 
