Installation
==============

There are two main ways to install Gym-TORAX:

1. Using pip from the Python Package Index (PyPI).

2. Installing it manually by cloning the GitHub repository.

.. In addition, TORAX must be installed separately, as described below.

Installing with pip (PyPI)
--------------------------------

The simplest method is to install Gym-TORAX directly from PyPI using ``pip``.
Run the following command in your terminal:

.. code-block:: bash

    pip install gym-torax

.. All dependencies (numpy, scipy and ply) will be automatically installed and the package should be ready for use.

Manual Installation
---------------------

The git repository can be found `here <https://github.com/antoine-mouchamps/gymtorax>`_. 
The repository can be cloned by typing the following commands in a terminal window:

.. code-block:: bash

   git clone https://github.com/antoine-mouchamps/gymtorax

Then, a local installation can be performed by typing the following commands in a 
terminal window:

.. code-block:: bash

    pip install .

If you only want Gym-TORAX as an uninstalled package, installing the requirements can 
be performed by typing the following commands:

.. code-block:: bash

   pip install -r requirements.txt

Testing
--------

To verify your installation is working correctly, you can run the test suite using pytest:

.. code-block:: bash

   pytest

This will run all the tests in the repository and ensure that Gym-TORAX is properly installed and functioning.
