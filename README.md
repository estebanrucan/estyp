# ESTYP: Extended Statistical Toolkit Yet Practical

[![Downloads](https://static.pepy.tech/badge/estyp)](https://pepy.tech/project/estyp) [![PyPI version](https://badge.fury.io/py/estyp.svg)](https://badge.fury.io/py/estyp) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Contributions](https://img.shields.io/badge/Contributions-welcome-blue.svg)](https://github.com/estebanrucan/estyp/issues) [![Chilean](https://img.shields.io/badge/Made_in-%F0%9F%87%A8%F0%9F%87%B1_Chile-blue.svg)](https://es.wikipedia.org/wiki/Chile)

## Description

This library is a collection of statistical functions for Python.

Actually, the name comes from the way my friends call me (Esti), plus "p" which is the initial of `python`.

## Changelog 

### V0.4.1

* Bug fixes in `linear_model.LogisticRegression()` class.
* Added downloads badge to README.
* Changed `sklearn>=1.2.1` to `scipy>=1.3.0` as a depedency of the library.

### V0.4.0

* Added `testing.prop_test()` function to perform a test of proportions.
* Added `testing.CheckModel()` class to perform linear regression assumptions checking.
* Added badges to README.
* Minor changes in README.

### V0.3.0

* Changed `scipy>=1.11.1` to `scipy>=1.10.1` as a depedency of the library.
* New modularization of the functions in the `linear_model` module.
* Added `linear_model.stepwise.forward_selection()` function to perform forward variable selection based in p-values.
* Added `testing.nested_models_test()` function to perform nested models testing.
* Added option to specity aditional parameters of the model like `kwargs` in `linear_model.stepwise.forward_selection()` and `linear_model.stepwise.both_selection()` functions.  
* Minor changes in README.

### V0.2.5

* Added `scipy>=1.11.1` as a depedency of the library.
* New modularization of the functions in the `testing` module.
* R like documentation in the `testing.var_test()` function.
* Added `testing.t_test()` function to perform t-test like in software R.

## Features

* `linear_model.LogisticRegression()`: This class implements a logistic regression model. It inherits from the `LogisticRegression()` class from `scikit-learn`, but adds additional methods for calculating confidence intervals, p-values, and model summaries like `Logit` class in `statsmodels`.
* `linear_model.stepwise.both_selection()`: This function performs both forward and backward variable selection using the Akaike Information Criterion (AIC). 
* `linear_model.stepwise.forward_selection()`: This function performs forward variable selection based on p-values.
* `testing.CheckModel()`: This class provides methods to test the assumptions of the linear regression model., inspired by the `performance::check_model()` function of the R software.
* `testing.t_test()`: Performs one and two sample t-tests on groups of data. This function is inspired by the `t.test()` function of the R software.
* `testing.var_test()`: Performs an F test to compare the variances of two samples from normal populations. This function is inspired by the `var.test()` function of the R software.
* `testing.prop_test()`: it can be used for testing the null that the proportions (probabilities of success) in several groups are the same, or that they equal certain given values. This function is inspired by the `prop.test()` function of the R software.
* `testing.nested_models_test()`: Performs a nested models test to compare two nested models using deviance criterion.

## Installation

To install this library, you can use PyPI:

```bash
pip install estyp
```

## License

This library is under the MIT license.

## Contact

If you have any questions about this library, you can contact me at [errucan@gmail.com](mailto:errucan@gmail.com).