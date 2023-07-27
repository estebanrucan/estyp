# ESTYP: Extended Statistical Toolkit Yet Practical

## Description

This library is a collection of statistical functions for Python.

Actually, the name comes from the way my friends call me (esty), plus the "p" which is the initial of `python`.

## Changelog 

### V0.3.0

* Changed `scipy>=1.11.1` to `scipy>=1.10.1` as a depedency of the library.
* New modularization of the functions in the `linear_model` module.
* Added `linear_model.stepwise.forward_selection()` function to perform forward variable selection based in p-values.
* Added `testing.nested_models_test()` function to perform nested models testing.
* Added option to specity aditional parameters of the model like `kwargs` in `linear_model.stepwise.forward_selection()` and `linear_model.stepwise.both_selection()` functions.  
* Minor changes in the README.

## Functions

* `linear_model.LogisticRegression()`: This class implements a logistic regression model. It inherits from the `LogisticRegression()` class from `scikit-learn`, but adds additional methods for calculating confidence intervals, p-values, and model summaries like `Logit` class in `statsmodels`.
* `linear_model.stepwise.both_selection()`: This function performs both forward and backward variable selection using the Akaike Information Criterion (AIC). 
* `linear_model.stepwise.forward_selection()`: This function performs forward variable selection based on p-values.
* `testing.t_test()`: Performs one and two sample t-tests on groups of data. This function is inspired by the `t.test()` function of the software R.
* `testing.var_test()`: Performs an F test to compare the variances of two samples from normal populations. This function is inspired by the `var.test()` function of the software R.
* `testing.nested_models_test()`: Performs a nested models test to compare two nested models using deviance criterion.

# Installation

To install this library, you can use pip:

```bash
pip install estyp
```

## License

This library is under the MIT license.

## Contact

If you have any questions about this library, you can contact me at [errucan@gmail.com](mailto:errucan@gmail.com).