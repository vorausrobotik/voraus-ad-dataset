# voraus-AD Dataset

This is the official repository to the IEEE Transactions on Robotics paper "[The voraus-AD Dataset for Anomaly Detection in Robot Applications]()" by Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn, and Bastian Wandt.

We introduce the voraus-AD dataset, a novel dataset for anomaly detection in robotic applications as well as an unsupervised method that is able to find anomalies on robotic machine data without having some of them in the training set.

[**Download the Dataset 100 Hz** (~1,1 GB Disk / ~2.5 GB RAM)](https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet)

[**Download the Dataset 500 Hz** (~5.3 GB Disk / ~12.5 GB RAM)](https://media.vorausrobotik.com/voraus-ad-dataset-500hz.parquet)

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

Make sure to execute the tests before training. The test `test_train` may take a few minutes depending on your setup.

```shell
pytest
```

The [train.py](train.py) is best entrypoint to this repository, it contains the configuration, training and validation steps. The default configuration will run a training with paper-given parameters on the provided voraus-AD dataset.

To start the training, just run [train.py](train.py)! If training on the voraus-AD data does not lead to an AUROC greater 0.9, something seems to be wrong. Don't be worried if the loss is negative. The loss reflects the negative log likelihood which may be negative.
Please report us if you have issues when using the code.

```shell
python train.py
```

## Data

Our voraus-AD dataset contains the machine data of a robot arm performing a vision based pick-and-place application. Check out the [video](https://media.vorausrobotik.com/voraus-ad-dataset.mp4) of the application.

Set the variable `DATASET_PATH` in [train.py](train.py) to the path of the downloaded dataset file.
The variable `configuration` contains the training configuration as well as the **hyperparameters** of the model. The paper describes all the configuration parameters in detail.  


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

and execute all checks using the following command:

```shell
tox
```

## Credits

Some code of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.


## Citation

Please cite our paper in your publications if it helps your research.

TODO: Not public yet.


## License Notices

This project is licensed under the [MIT License](https://opensource.org/license/mit/).
