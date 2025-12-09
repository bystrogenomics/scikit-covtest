.. _covtest-methods:

Covariance hypothesis testing methods
=====================================

This page documents the covariance hypothesis testing methods implemented
in :mod:`covtest.methods`. These include one-sample identity and sphericity
tests, proportionality tests, and two-sample or multi-sample equality tests
for covariance matrices.

All procedures are designed for use in both classical and high-dimensional
settings, with several trace-based, likelihood-based, and robust variants.


Identity covariance tests
-------------------------

.. currentmodule:: covtest.methods.hypothesis_identity

.. automodule:: covtest.methods.hypothesis_identity
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ahmad2015_identity
   ledoit_wolf_identity
   nagao_identity
   srivastava_2005_identity
   tyler_identity
   fisher_single_sample
   srivastava2011_single_sample


Sphericity tests
----------------

.. currentmodule:: covtest.methods.hypothesis_spherical

.. automodule:: covtest.methods.hypothesis_spherical
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   bartlett_sphericity_test
   john_sphericity
   srivastava_2005_sphericity
   sk_test
   muirhead_sphericity_lrt
   czz_sphericity_test
   hallin_rank_sphericity_test


Proportionality tests
---------------------

.. currentmodule:: covtest.methods.hypothesis_proportionality

.. automodule:: covtest.methods.hypothesis_proportionality
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   flury_proportionality_test
   bartlett_adjusted_proportionality_test
   proportionality_test_LZ
   proportionality_test_signs
   proportionality_plrt
   proportional_cov_test_tsukuda


Two-sample and multi-sample covariance tests
--------------------------------------------

.. currentmodule:: covtest.methods.hypothesis_two_sample

.. automodule:: covtest.methods.hypothesis_two_sample
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ahmad_2015_two_sample
   boxm_test
   ishii_two_sample
   schott_2001
   srivastava_yanagihara_two_sample
   srivastava_two_sample_2007
   wald_two_sample
   tyler_two_sample
   cai_2013_two_sample
   he_2018_two_sample
   chang2016
   schott2007
   ding2023_two_sample

