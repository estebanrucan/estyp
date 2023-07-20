# ESTYP: Extended Statistical Toolkit Yet Practical

## Description

This library is a collection of statistical functions for Python. It includes a function for performing stepwise with AIC criterion and a function for the F-ratio test.

Actually the name comes from the way my friends call me (esty), plus the "p" which is the initial of `python`.

## Functions

`linear_model.LogisticRegression()`: This class implements a logistic regression model. It inherits from the `sklearn.linear_model.LogisticRegression` class, but adds additional methods for calculating confidence intervals, p-values, and model summaries.
`linear_model.stepwise.both_selection()`: This function performs both forward and backward variable selection using the Akaike Information Criterion (AIC). 
`testing.var_test()`: Performs an F-ratio test to compare the variances of two samples. The output was inspired by the `var_test()` function in `R`.

# Installation

To install this library, you can use pip:

```bash
pip install estyp
```

## License

This library is under the MIT license.

## Contact

If you have any questions about this library, you can contact me at [errucan@gmail.com](mailto:errucan@gmail.com).