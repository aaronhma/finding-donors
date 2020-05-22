# Intro to ML with TensorFlow: Finding Donors
![Mark stale issues and pull requests](https://github.com/aaronhma/finding-donors/workflows/Mark%20stale%20issues%20and%20pull%20requests/badge.svg?branch=master)
![Website Status](https://img.shields.io/badge/website-passing-brightgreen)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Project Status](https://img.shields.io/badge/project-in--review-orange)

This repository contains code submitted for Udacity's Intro to ML with TensorFlow - Project 1: Finding Donors

## Contents
<!-- MarkdownTOC depth=4 -->
- [Finding Donors](https://github.com/aaronhma/finding-donors/)
  - [Getting Started](https://github.com/aaronhma/finding-donors#getting-started)
  - [Writeup](#writeup)
    - [Placeholder](#placeholder)
  - [Files](#files)
  - [Libraries Used](#libraries)
  - [Contributing](#guidelines)
  - [License](#copyright)
<!-- /MarkdownTOC -->

<a name = "setup" />

## Getting Started
1. Clone this repository.
```bash
# With HTTPS:
$ git clone https://github.com/aaronhma/finding-donors.git aaronhma-finding-donors
# Or with SSH:
$ git clone git@github.com:aaronhma/finding-donors.git aaronhma-finding-donors
```
2. Get into the repository.
```bash
$ cd aaronhma-finding-donors
```

3. Install the required dependencies.
```bash
$ pip3 install -r requirements.txt
```

4. Start the Jupyter Notebook/Jupyter Lab or click [here to access the server](http://localhost:8888).
```bash
# For jupyter notebook:
$ jupyter notebook
# Then, go to http://localhost:8888/tree?token=<YOUR TOKEN HERE>
# ============================
# For jupyter lab:
$ jupyter lab
# Then, go to http://localhost:8888/lab?token=<YOUR TOKEN HERE>
```

5. Enjoy the project!

<a name = "writeup" />

## Writeup
### 0. Preparation Work
#### Installing Packages
Before we begin the project, we must install all the packages. Follow along to get your own copy of the packages used in this project.

```bash
# If you're in the Terminal:
$ pip install -r requirements.txt
# Or if you're in Jupyter:
!pip install -r requirements.txt
```

#### Loading Packages
After installing the packages we must load it for the project.

```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs
```

### 1. Exploring the Data
#### Install the data
In order to install the data, please sign in to your classroom to install it.

#### Load the data
Assuming you've installed the data, we'll load the data with [Pandas](pandas.pydata.org).

```python
# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))
```

#### Data Exploration
Now, it's time to actually start the project!

```python
# TODO: Total number of records
n_records = None

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = None

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = None

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = None

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
```

<a name = "files" />

## Files

Here's the current, up-to-date, file structure for this project:

```
|____CODE_OF_CONDUCT.md
|____LICENSE
|____requirements.txt
|____finding_donors.ipynb
|____README.md
|____.gitignore
|____.github
| |____FUNDING.yml
| |____workflows
| | |____stale.yml
| | |____greetings.yml
|____visuals.py
|____SECURITY.md
```

<a name = "libraries" />

## Libraries Used

The libraries used for this project can be found [here](https://github.com/aaronhma/finding-donors/blob/master/requirements.txt).

<a name = "guidelines" />

## Contributing
Contributions are always welcome!

<a name = "copyright" />

## License
The MIT License is used for this project, which you can read [here](https://github.com/aaronhma/aitnd-momentum-trading/blob/master/LICENSE):

```
MIT License

Copyright (c) 2020 - Present Aaron Ma,
Copyright (c) 2020 - Present Udacity, Inc.
All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

In short, here's the license:
```
A short and simple permissive license with conditions only requiring
preservation of copyright and license notices.
Licensed works, modifications, and larger works may be
distributed under different terms and without source code.
```

| Permissions                      | Allowed?           |
| -------                          | ------------------ |
| Commercial use                   | :white_check_mark: |
| Modification                     | :white_check_mark: |
| Re-distribution (with license)   | :white_check_mark: |
| Private use (with license)       | :white_check_mark: |
| Liability                        | :x:                |
| Warranty                         | :x:                |

**The checkmarked items are allowed with the condition of the original license and copyright notice.**

For more information, click [here](https://www.copyright.gov/title17/title17.pdf).
