.. ESTYP Documentation documentation master file, created by
   sphinx-quickstart on Sun Aug  6 23:02:06 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ESTYP Documentation's documentation!
===============================================

Hello, my name is Esteban Rucán and I glad to share this project with you. I'm a data scientist from Chile and I'm interested in the development of tools for the analysis of data. Always I'm looking for new ways to improve my skills and I think that the best way to do it is sharing my knowledge with others. I hope that this project will be useful for you.

ESTYP (Extended Statistical Toolkit Yet Practical) is a Python library that serves as a multifaceted toolkit for statistical analysis:

- The `testing` module encompasses a wide range of statistical tests, including t-tests, chi-squared tests, and correlation tests, providing robust methods for data comparison and validation. These functions are inspired by their analogs in R software, ensuring user-friendliness. 

- In the `linear_model` module, users can find functionalities related to logistic regression, including variable selection techniques and additional methods for calculating confidence intervals and p-values. This module enhances the capabilities of traditional logistic regression analysis in `scikit-learn` and ensuring convergence in cases that `statmodels` can't afford. 

- The cluster module is designed to assist in clustering analysis, offering tools to identify the optimal number of `clusters` using methods like the elbow or silhouette techniques. 

Together, these modules form a comprehensive and practical statistical toolkit that caters to various analytical needs.

This documentation is also powered by `Jupyter`. You can start using the ESTYP library by `installing it <./instalation.html>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   instalation
   getting_started
   cluster
   linear_model
   testing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
