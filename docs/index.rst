scikit-covtest documentation
============================

scikit-covtest is a Python package for high dimensional covariance matrix
testing, with a focus on both classical and high-dimensional tests

It provides statistically principled tests for:

- Identity and sphericity of covariance matrices
- Proportionality between two covariance matrices
- Two-sample equality testing in high dimensions
- Multiple testing control for large test families

The library is designed for:

- Statistical genomics and gene expression analysis
- High dimensional machine learning diagnostics
- Simulation driven theoretical validation
- Large scale hypothesis testing pipelines

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install scikit-covtest

Or install from source:

.. code-block:: bash

   git clone https://github.com/bystrogenomics/scikit-covtest.git
   cd scikit-covtest
   pip install .

Quickstart
----------

Minimal example using a covariance identity test:

.. code-block:: python

   import numpy as np
   from covtest.methods import hypothesis_identity as hi

   rng = np.random.default_rng(0)
   X = rng.normal(size=(200, 50))

   result = hi.test_identity_T2(X)
   print("Statistic: ", result['stat'])
   print("p-value: ", result['p_value'])

User guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/methods
   user_guide/multiple_testing
   user_guide/diagnostics
   user_guide/simulation

API reference
-------------

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/covtest.datasets
   api/covtest.diagnostics
   api/covtest.methods
   api/covtest.multiplicity
   api/covtest.plotting
   api/covtest.simulation
   api/covtest.testing
