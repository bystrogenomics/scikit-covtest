.. _covtest-plotting:

Plotting utilities
==================

This page documents the plotting utilities implemented in :mod:`covtest.plotting`.
These include functions for visualizing power curves, p-value distributions,
and covariance matrix structures.

Power analysis plots
--------------------

.. currentmodule:: covtest.plotting.alternative

.. automodule:: covtest.plotting.alternative
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot_power_curve
   plot_power_heatmap
   plot_mean_pvalue
   plot_pvalue_distributions
   plot_power_curves_multi_alpha

Null distribution diagnostics
-----------------------------

.. currentmodule:: covtest.plotting.null

.. automodule:: covtest.plotting.null
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot_pval_histogram
   plot_pval_qq
   plot_pval_ecdf
   plot_pval_cumdev
   plot_pval_obs_vs_exp
   plot_pval_survival
   plot_pval_running_mean
   plot_pval_zdist
   plot_pval_calibration
   qq_plot_log_p
   pvalue_histogram
   plot_pval_diagnostics
   plot_pvalue_diagnostics_grid

Structure visualization
-----------------------

.. currentmodule:: covtest.plotting.structure

.. automodule:: covtest.plotting.structure
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot_eigen_scree
   plot_matrix_structure
