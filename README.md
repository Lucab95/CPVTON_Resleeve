CPVTON resleeve assesment
==============================
Assesment for Resleeve, to inference and test new images using [CPVTON+](https://github.com/minar09/cp-vton-plus).


Installation:
--------------
This projects uses git lfs. </br>
After cloning the repository, run `git lfs install` to install git-lfs.
Then run `git lfs pull` to download the files. </br>
For any problem to dowload the checkpoints,model and the openposi api, refer to the mega link below. </br>
https://mega.nz/folder/h49ymDTb#ydJTDXkEv7J9cyy7Ye7Oeg

the packages are in the requirements.txt file. A virtual environment is also available in the mega folder.


Data preparation
=============================

To test on new images, the images need to be added in the ```data/raw/image``` and ```data/raw/cloth``` folder. In addition the file test_wild_pairs.txt needs to be added in the ```data/folder```. The file contains the image names and the cloth names.


Example:
--------
image_name1.jpg cloth_name1.jpg </br>
image_name2.jpg cloth_name5.jpg


Testing
=======

To test the model, run the following command: </br>
python test_extdata.py </br>
No parameters are required because they are already set as default. </br>
Although, the parameters can be changed as per the need.


Attributes:
-------------------
```
--gpu_ids : list of integers, which GPUs to use. default: 1
--workers : number of workers for data loading. default: 1
--batch_size : size of batch. default: 4
--dataroot : path to the data folder. default: 'data'
--datamode : folder in wich the preprocessed images will be saved. default: 'processed'
--stage : model to execute. However, the models are applied in sequence, first GMM, then the results is used for TOM. default: 'GMM'
--fine_width : width of the input image. default: 192
--fine_height : height of the input image. default: 256
--radius : radius of gaussian kernel default: 5
--grid_size : size of the grid default: 5
--result_dir : path to the result folder. default: 'result'
--checkpoint_GMM: path to the checkpoint of GMM. default: 'checkpoints/GMM/gmm_final.pth'
--checkpoint_TOM: path to the checkpoint of TOM. default: 'checkpoints/TOM/tom_final.pth'
--display_count : number of steos to receive prints from the model. default: 5
--shuffle : whether to shuffle the data. default: True
```
Possible issues:
------------------
Openpose_api has been tested only on Windows. If you are receiving problems, be sure that the path inside openpose_api\keypoints_from_images.py
```
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '\Release;' +  dir_path + '/bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
```

Additionally, you might also check

```
    params["model_folder"] = "models/"
```




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
