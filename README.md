# voraus-AD Dataset

This is the official repository to the paper [**"The voraus-AD Dataset for Anomaly Detection in Robot Applications"**](https://arxiv.org/pdf/2311.04765.pdf) by Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn, and Bastian Wandt which is accepted to IEEE Transactions on Robotics and will be officially published soon.

We introduce the **voraus-AD dataset**, a novel dataset for **anomaly detection** in robotic applications as well as an unsupervised method **MVT-Flow** which finds anomalies on **time series of robotic machine data** without having some of them in the training set.

[**Download the Dataset 100 Hz** ](https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet)    
(~1,1 GB Disk / ~2.5 GB RAM) - used in this repository

[**Download the Dataset 500 Hz**](https://media.vorausrobotik.com/voraus-ad-dataset-500hz.parquet)    
(~5.3 GB Disk / ~12.5 GB RAM)

**Please note:** The datasets in both the 100 Hz and 500 Hz variants are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0).

## Getting Started

You will need [Python 3.9](https://www.python.org/downloads/) and the packages specified in requirements.txt. We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) and installing the packages there.

Install packages with:

```shell
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configure and Run

Set the variable `DATASET_PATH` in [train.py](train.py) to the path of the downloaded dataset file.
The variable `configuration` contains the training configuration as well as the hyperparameters of the model. The paper describes all the configuration parameters in detail. Make also sure to execute the tests before training. The test `test_train` may take a few minutes depending on your setup.

```shell
pytest
```

The [train.py](train.py) is entrypoint to this repository, it contains the configuration, training and validation steps for our model. The default configuration will run a training with **paper-given parameters** on the provided voraus-AD dataset (@100 Hz).
To start the training, just run [train.py](train.py)! 

```shell
python train.py
```

If training on the voraus-AD data does not lead to an AUROC greater 0.9, something seems to be wrong. Don't be worried if the loss is negative. The loss reflects the negative log likelihood which may be negative.
Please report us if you have issues when using the code.


## Devlopment

We are using the following tools during development:

- [isort](https://github.com/pycqa/isort/) for import sorting
- [black](https://github.com/psf/black) for code formatting
- [mypy](https://github.com/python/mypy) for static typing
- [pylint](https://github.com/pylint-dev/pylint) for static code analysis (linting)
- [pydocstyle](https://github.com/PyCQA/pydocstyle) for Docstring style checking 
- [pytest](https://github.com/pytest-dev/pytest/) for (unit) testing
- [tox](https://github.com/tox-dev/tox) for test automation

Before commiting make sure to format your code with:

```shell
isort .
black .
```

And execute all checks using the following command:

```shell
tox
```

**Note:** Running **tox** the first time takes a few minutes since tox creates new virtual environments for linting and testing. The following **tox** executions are much faster.

## Credits

Some code of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.


## Citation

Please cite our paper in your publications if it helps your research.

    @article { BroRud2023,
      author = {Jan Thie{\"s} Brockmann and Marco Rudolph and Bodo Rosenhahn and Bastian Wandt},
      title = {The voraus-AD Dataset for Anomaly Detection in Robot Applications},
      journal = {Transactions on Robotics},
      year = {2023},
      month = nov
    }


## License Notices

The **content of this repository** is licensed under the [MIT License](https://opensource.org/license/mit/).   
The **datasets** are licensed under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
