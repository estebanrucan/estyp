# ESTYP: Extended Statistical Toolkit Yet Practical

[![Downloads](https://static.pepy.tech/badge/estyp)](https://pepy.tech/project/estyp) [![](https://github.com/estebanrucan/estyp/actions/workflows/test.yml/badge.svg)](https://github.com/estebanrucan/estyp/actions/workflows/test.yml) [![Documentation Status](https://readthedocs.org/projects/estyp/badge/?version=latest)](https://estyp.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/estyp.svg)](https://badge.fury.io/py/estyp) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Contributions](https://img.shields.io/badge/Contributions-welcome-blue.svg)](https://github.com/estebanrucan/estyp/issues) [![Chilean](https://img.shields.io/badge/Made_in-%F0%9F%87%A8%F0%9F%87%B1_Chile-blue.svg)](https://es.wikipedia.org/wiki/Chile)

## Description

ESTYP (Extended Statistical Toolkit Yet Practical) is a Python library that serves as a multifaceted toolkit for statistical analysis. The `testing` module encompasses a wide range of statistical tests, including t-tests, chi-squared tests, and correlation tests, providing robust methods for data comparison and validation. In the `linear_model` module, users can find functionalities related to logistic regression, including variable selection techniques and additional methods for calculating confidence intervals and p-values. This module enhances the capabilities of traditional logistic regression analysis. The cluster module is designed to assist in clustering analysis, offering tools to identify the optimal number of `clusters` using methods like the elbow or silhouette techniques. Together, these modules form a comprehensive and practical statistical toolkit that caters to various analytical needs. 

Actually, the name comes from the way my friends call me (Esti), plus "p" which is the initial of `python`.

## Installation

To install this library, you can use PyPI:

```bash
pip install estyp
```

Also, you can install it from the source code:

```bash
git clone https://github.com/estebanrucan/estyp.git
cd estyp
pip install -e .
```

## Documentation

You can have a friendly introduction to this library in the [documentation](https://estyp.readthedocs.io/en/latest/).

## Changelog

You can see the full changelog [here](./CHANGELOG.md).

## Features

### `testing` module

* `testing.CheckModel()`: This class provides methods to test the assumptions of the linear regression model., inspired by the `performance::check_model()` function of the R software.
* `testing.t_test()`: Performs one and two sample t-tests on groups of data. This function is inspired by the `t.test()` function of the R software.
* `testing.var_test()`: Performs an F test to compare the variances of two samples from normal populations. This function is inspired by the `var.test()` function of the R software.
* `testing.prop_test()`: it can be used for testing the null that the proportions (probabilities of success) in several groups are the same, or that they equal certain given values. This function is inspired by the `prop.test()` function of the R software.
* `testing.chisq_test()`: Performs a chi-squared test of independence of variables in a contingency table. This function is inspired by the `chisq.test()` function of the R software.
* `testing.cor_test()`: Performs a correlation test with Pearson, Spearman or Kendall method. This function is inspired by the `cor.test()` function of the R software.
* `testing.nested_models_test()`: Performs a nested models test to compare two nested models using deviance criterion.
* `testing.dw_test()`: Performs the Durbin-Watson test for autocorrelation of disturbances (includes a p-value). Inspired by the `lmtest::dwtest()` function of the R software.

### `linear_model` module

* `linear_model.LogisticRegression()`: This class implements a logistic regression model. It inherits from the `LogisticRegression()` class from `scikit-learn`, but adds additional methods for calculating confidence intervals, p-values, and model summaries like `Logit` class in `statsmodels`.
* `linear_model.Stepwise()`:  Provides a implementation to add or remove predictors based on their significance, AIC or BIC in a model.


### `cluster` module

* `cluster.NClusterSearch`: A helper class to identify the optimal number of clusters for clustering algorithms with elbow or silhuette methods.

## License

This library is under the MIT license.

## Contact

If you have any questions about this library, you can contact me at [LinkedIn](https://www.linkedin.com/in/estebanrucan/).