# Streaming Inference for Infinite Feature Models

This code corresponds to our AISTATS 2021 submission.

![](01_prior/results/a=10.78_b=2.3/ibp_recursion_a=10.78_b=2.3.png)

## Setup

After cloning the repository, create a virtual environment for Python 3:

`python3 -m venv ribp`

Then activate the virtual environment:

`source ribp/bin/activate`

Ensure pip is up to date:

`pip install --upgrade pip`

Then install the required packages:

`pip install -r requirements.txt`

We did not test Python2, but Python2 may work.


## Running

Each experiment has its own directory, each containing a `main.py` that creates a `plots`
subdirectory (e.g. `exp_00_ibp_prior/plots`) and then reproduces the plots in the paper. Each 
`main.py` should be run from the repository directory e.g.:

`python3 exp_00_ibp_prior/main.py`

## TODO
- Results on synthetic linear-gaussian data
  - Plot comparison of ground-truth features and inferred features
  - Plot comparison of ground-truth indicators and inferred indicators
- MNIST or Omniglot
  - Implement MNIST comparison of classes based on inferred features
  - PCA MNIST
  - Implement Omniglot
- Landmark Dataset
  - linear gaussian -> NMF
- Maybe more interesting: show different features under different model
  assumptions (e.g. FA vs NMF)
- Derive
  - Factor Analysis
  - Non-negative Matrix Factorization
- Implement
  - Factor Analysis
  - Non-negative Matrix Factorization
- Implement IBP baselines
  - One of Paisley & Carin's papers
  - Particle Filtering (Wood and Griffiths)

## Contact

Questions? Comments? Interested in collaborating? Open an issue or 
email Rylan Schaeffer (rylanschaeffer@gmail.com) and cc Yilun Du
(yilundu@gmail.com) and Ila Fiete (fiete@mit.edu).
