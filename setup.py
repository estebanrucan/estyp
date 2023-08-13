from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name                          = "estyp",
    version                       = "0.7.0",
    author                        = "estebanrucan",
    author_email                  = "errucan@gmail.com",
    description                   = "Extended Statistical Toolkit Yet Practical",
    long_description              = readme,
    long_description_content_type = "text/markdown",
    url                           = "https://github.com/estebanrucan/estyp",
    project_urls                  = {
        "Bug Tracker": "https://github.com/estebanrucan/estyp/issues",
        "Homepage": "https://github.com/estebanrucan/estyp",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages         = find_packages(),
    python_requires  = ">=3.9.12",
    install_requires = [
        "numpy >= 1.22.3",
        "scikit-learn >= 1.3.0",
        "matplotlib >= 3.4.3",
        "patsy >= 0.5.3",
        "statsmodels >= 0.13.5",
        "scipy >= 1.10.1",
        "kmodes >= 0.12.2",
        "tqdm >= 4.65.0"
    ],
    license              = "MIT",
    include_package_data = True,
)
