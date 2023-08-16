# CHANGELOG

### V0.8.0

* Added `testing.dw_test()` to perform Durbin-Watson test with an aproximatted p-value.
* Updated the `testing.CheckModel` class to use the `testing.dw_test()` function.
* Fixed some displaying problems in the documentation.
* Now some functions of the `testing` module forces to use named parameters instead of positional ones.
* Replaced ":" for "=" in the `TestResults` class display.

### V0.7.0

* Added a new library import message that derives to documentation.
* Changed `LogisticRegresion.deviance_residuals()` method to a property.
* Fixed AIC calculation in `LogisticRegression` class.
* Added deviance and deviance_residuals properties from `LogisticRegression` class to documentation.
* Added Dockerfile to create a docker image to build the package.
* Added tests to check if functionalities works properly. The tests are in the `test` folder and runs with GitHub Actions in every push.
* Changed pyproject.toml to a setup.py file.

### V0.6.0

* I'm glad to announce that this library have it own documentation in [Read the Docs](https://estyp.readthedocs.io/en/latest/).
* All functions and classes docstrings have a link that redirects to it web documentation.

### V0.5.1 (Preparing for docs)

* Added CHANGELOG.md file.
* Added .readthedocs.yml file.
* Changed library description in README.md.
* All functions and classes have a reviewed docstring.
* Added `tqmd >= 4.65.0` as a depedency of the library.
* Fixed some bugs in `testing.TestResults()` class.
* Fixed some bugs in `cluster.NClusterSearch()` class.
* Fixed some bugs in `linear_model.LogisticRegression()` class.
* Now `linear_model.LogisticRegression()` has a `deviance` attribute.
* `testing.nested_models_test()` function now accepts `linear_model.LogisticRegression()` instances.
* `linear_model.stepwise.forward_selection()` function now accepts `linear_model.LogisticRegression()` instances.
* `linear_model.stepwise.both_selection()` and `linear_model.stepwise.backward_selection()` functions now accepts `verbose`, `formula_kwargs` and `fit_kwargs`  parameters.

### V0.5.0

* Added `testing.chisq_test()` function to perform a chi-squared test.
* Added `testing.cor_test()` function to perform a correlation test.
* Added `cluster.NClusterSearch()` class to identify the optimal number of clusters for clustering algorithms with elbow or silhuette methods.
* Added `kmodes >= 0.12.2` as a depedency of the library.
* Added `__version__` atribute to the library.
* Changed method displaying in `TestResults` class.
* Minor changes in README.

### V0.4.1

* Bug fixes in `linear_model.LogisticRegression()` class.
* Added downloads badge to README.
* Changed `sklearn>=1.2.1` to `sklearn>=1.3.0` as a depedency of the library.

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
