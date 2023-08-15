Instalation
===========

You can install the ESTYP library using PyPI:

.. code-block:: console

    pip install estyp

Also, you can install it from the source code:

.. code-block:: console
    
    git clone https://github.com/estebanrucan/estyp.git
    cd estyp
    pip install -e .


The lastest version of the library in PyPI is:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import estyp

.. jupyter-execute::
    :hide-code:

    print(estyp.__version__)

The library was tested in the versions 3.9, 3.10 and 3.11 of Python. The following dependencies will also be installed:

-  "numpy >= 1.22.3"
-  "scikit-learn >= 1.3.0"
-  "matplotlib >= 3.4.3"
-  "patsy >= 0.5.3"
-  "statsmodels >= 0.13.5"
-  "scipy >= 1.10.1"
-  "kmodes >= 0.12.2"
-  "tqdm >= 4.65.0"

You can view the last changes of the library in the `CHANGELOG.md <https://github.com/estebanrucan/estyp/blob/main/CHANGELOG.md>`_ file from the GitHub repository.

Once installed, you can import the library using:

.. code-block:: python

    import estyp


Visit the `Getting Started <getting_started.hmtl>`_ section for more information.