# CHANGELOG

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