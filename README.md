# Accident Severity Prediction<img src='https://avatars.githubusercontent.com/u/93612370' align='right' width='180' height='104'>
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/UTK-ML-Dream-Team/accident-severity-prediction/master/LICENSE)

## About  <a name = "about"></a>
**[Project Board](https://github.com/UTK-ML-Dream-Team/accident-severity-prediction/projects/1)**, 
**[Current Issues](https://github.com/UTK-ML-Dream-Team/accident-severity-prediction/issues)**, 
**[Assignment](http://web.eecs.utk.edu/~hqi/cosc522/project/proj-final.htm)**

Group 2 - Final Project for the UTK Machine Learning Course (COSC-522)

### Libraries Overview <a name = "lib_overview"></a>

All the libraries are located under [\<project root\>/project_libs](project_libs)
- [\<project root\>/project_libs/project](project_libs/project): This project's code
- [\<project root\>/project_libs/configuration](project_libs/configuration): Class that creates config objects from yml files
- [\<project root\>/project_libs/fancy_logger](project_libs/fancy_logger): Logger that can be used instead of prints for text formatting (color, bold, underline etc)

### Where to put the code  <a name = "#putcode"></a>
- [Main Notebook](main.ipynb)
- Place the preprocessing functions/classes in [\<project root\>/project_libs/project/preprocessing.py](project_libs/project/preprocessing.py)
- The models in [\<project root\>/project_libs/project/models.py](project_libs/project/models.py)
- Any plotting related functions in [\<project root\>/project_libs/project/plotter.py](project_libs/project/plotter.py)

**The code is reloaded automatically. Any class object needs to reinitialized though.** 

## Table of Contents

+ [About](#about)
  + [Libraries Overview](#lib_overview)
  + [Where to put the code](#putcode)
+ [Prerequisites](#prerequisites)
+ [Bootstrap Project](#bootstrap)
+ [Running the code using Jupyter](#jupyter)
      + [Configuration](#configuration)
      + [Local Jupyter](#local_jupyter)
      + [Google Collab](#google_collab)
+ [Adding New Libraries](#adding_libs) 
+ [License](#license)

## Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python >= 3.8 and any Bash based shell (e.g. zsh) installed.
Having installed conda is also recommended.

```Shell

$ python3.8 -V
Python 3.8

$ echo $SHELL
/usr/bin/zsh

```

## Bootstrap Project <a name = "bootstrap"></a>

All the installation steps are being handled by the [Makefile](Makefile).

If you want to use conda run:
```Shell
$ make install

$ conda activate accident_severity_prediction

```

If you want to use venv run:
```Shell
$ make install env=venv
```

## Using Git <a name = "git"></a>

To push your local changes to remote repository:

2. For every file you changed do:
```Shell
$ git add path-to-file-1
$ git add path-to-file-2
# ...
``` 
2. Create a commit message
```Shell
$ git commit -m "My commit message"
```
2. Push your changes to GitHub
```Shell
$ git fetch
$ git pull
$ git commit -m "My commit message"
$ git push origin master
```

To pull changes from GitHub
```Shell
$ git pull
```

## Using Jupyter <a name = "jupyter"></a>

### Modifying the Configuration <a name = "configuration"></a>

You may need to configure the yml file. There is an already configured yml file 
under [confs/prototype1.yml](confs/prototype1.yml).

### Local Jupyter <a name = "local_jupyter"></a>

First, make sure you are in the correct virtual environment:

```Shell
$ conda activate accident_severity_prediction

$ which python
/home/<your user>/anaconda3/envs/accident_severity_prediction/bin/python
```

To use jupyter, first run `jupyter`:

```shell
jupyter notebook
```
And open the [main.ipynb](main.ipynb).

### Google Collab <a name = "google_collab"></a>

Just Open this [Google Collab Link](https://colab.research.google.com/github/UTK-ML-Dream-Team/accident-severity-prediction/blob/main/main.ipynb).

## Adding New Libraries <a name = "adding_libs"></a>

If you want to add a new library (e.g. a Class) in the project you need to follow these steps:
1. Go to *"\<project root>/project_libs/project"*
2. Create a new python file inside it with a name like *my_module.py*
3. Paste your code inside it
4. Go to *project_libs/project/__init__.py*
7. Add the following line: ```from project_libs.project/<Module name> import *```
8. (Optional) Rerun `make install` or `python setup.py install` 

## License <a name = "license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


