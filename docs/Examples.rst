Examples
=========

.. `tutorial <https://torax.readthedocs.io/en/v1.0.3/tutorials.html>`_ is normally the exact scenario

This section illustrates how to use Gym-TORAX with practical cases.  
All examples are based on the ITER hybrid ramp-up scenario, provided with TORAX.  
This scenario, adapted from [Citrin_2010], consists of:

- a ramp-up phase (0–100 s) in *L-mode* (low confinement regime),
- followed by a nominal phase (100–150 s) in *H-mode* (high confinement regime).

We present three parts:

1. :doc:`Environment description <Iter env>` — details about the custom Gym-TORAX environment ``IterHybridEnv``.
2. :doc:`Physical validation <Physical validation>` — a direct comparison with the TORAX reference simulation.
3. :doc:`PI controller <Simple control>` — a first proof-of-concept control example.

.. toctree::
   :maxdepth: 1
   :hidden:

   Iter env
   Physical validation
   Simple control