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

   test_identity_T2
   ledoit_wolf_identity
   nagao_identity
   srivastava_2005_identity
   tyler_identity
   fisher_single_sample
   srivastava2011_single_sample
   one_sample_cov_test



Sphericity tests
----------------

.. currentmodule:: covtest.methods.hypothesis_spherical

.. automodule:: covtest.methods.hypothesis_spherical
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ahmad2015_sphericity_test
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
   ahmad_2022_proportionality_test


Two-sample and multi-sample covariance tests
--------------------------------------------

.. currentmodule:: covtest.methods.hypothesis_two_sample

.. automodule:: covtest.methods.hypothesis_two_sample
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ahmad_2017_two_sample
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
   two_sample_cov_test
   cai_liu_xia_2013_two_sample_test
   chang_2017_perturbation_max_test



Random Matrix Theory (RMT) distribution statistics
--------------------------------------------------

.. currentmodule:: covtest.methods.rmt_stat

.. automodule:: covtest.methods.rmt_stat
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dmp
   pmp
   qmp
   dtw
   ptw
   qtw
   d_wishart_max
   p_wishart_max
   q_wishart_max
   d_wishart_spike
   p_wishart_spike
   q_wishart_spike
   wishart_max_par
   wishart_spike_par
   d2
   mu2
   sigma2_2
   d2_hat

