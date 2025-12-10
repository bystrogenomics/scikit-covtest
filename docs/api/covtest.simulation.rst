.. _covtest-simulation:

Simulation utilities
====================

This page documents the simulation utilities implemented in :mod:`covtest.simulation`.
These include functions for generating covariance matrices with specific structures
and generating multivariate data from various distributions.

Covariance matrix generation
----------------------------

.. currentmodule:: covtest.simulation.generate_covariances

.. automodule:: covtest.simulation.generate_covariances
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   sample_goe
   sample_cov_uniform_correlation
   generate_spectral_cov
   generate_toeplitz_cov
   generate_block_diagonal_cov
   generate_low_rank_cov
   generate_sparse_precision_cov
   generate_marchenko_pastur
   generate_spiked_covariance

Data generation
---------------

.. currentmodule:: covtest.simulation.generate_data

.. automodule:: covtest.simulation.generate_data
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   generate_heavy_tailed_samples
   generate_heavy_tailed_alternative
