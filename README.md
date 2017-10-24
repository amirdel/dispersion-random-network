# Modeling dispersion in porous media #

### What is this repository for? ###

This code is for modeling macro-dispersion in random porous networks with temporal Markov models.

* Markov models in time for simulation dispersion in random porous networks
* The stencil method and the extended stencil method
* An implementation of correlated CTRW for benchmarking results

For details about the ideas and models used in this code please refer to
[Temporal Markov Processes for Transport in Porous Media: Random Lattice Networks](https://arxiv.org/abs/1708.04173)


### How do I set it up? ###

* First clone this repository.
* Add the parent directory of py_dp to your Python path. In linux you will need to
add a line similar to `export PYTHONPATH=$PYTHONPATH:/PATH_TO_PARENT_OF_py_dp` to your `~/.bashrc` file.
* Make sure you have Python2.7 along with Matplotlib, Numpy, Scipy, Cython and pyamg.
All of these packages can be installed using pip. (e.g. `pip install numpy`).
* Complete Cython installations:
    - go to the dispersion directory: `cd py_dp/dispersion/`
    - compile the cython files using these commands:
        * `python setup_count.py build_ext --inplace`
        * `python setup_convert.py build_ext --inplace`

### Getting started with the code ###

After the setup steps you can run `sample_scripts/workflow_dispersion_in_random_network.py`.
This file includes all the steps necessary to generate the input data for a small 100x100
structured network, generating MC data, and model the data using Markov models.


