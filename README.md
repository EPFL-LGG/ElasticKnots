# ElasticKnots
A framework to simulate knotted elastic rods with self-contacts.

This is the codebase associated to the SIGGRAPH 2023 paper [Computational Exploration of Multistable Elastic Knots](https://infoscience.epfl.ch/record/302408).

To know more about our research, please visit the project [webpage](https://www.epfl.ch/labs/gcm/research-projects/elastic-knots/).

# Getting Started

## C++ Code Dependencies
To install the required dependencies, please follow the instructions from https://github.com/jpanetta/ElasticRods

## Obtaining and Building
Clone this repository *recursively* so that its submodules are also downloaded:

```
git clone git@github.com:EPFL-LGG/ElasticKnots.git --recurse-submodules
```

Build the C++ code and its Python bindings using cmake and your favorite build system. For example, with `ninja`:

```
cd ElasticKnots
mkdir build && cd build
cmake .. -GNinja
ninja
```

## Python Dependencies
The file `environment.yml` contains all the Python dependencies needed to run the demo Jupyter notebooks.
To create the conda environment `ElasticKnots` run:
```
conda env create -f environment.yml
```

# Running the Jupyter Notebooks
Launch Jupyter lab from the root directory:

```
jupyter lab
```

Now try opening and running an example notebook, e.g., `python/Demo-1-Simulation.ipynb` to test the simulation framework.
