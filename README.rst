============
pseudopeople
============

pseudopeople is a Python package that generates realistic simulated data about a
fictional United States population, designed for use in testing entity resolution
(record linkage) methods or other data science algorithms at scale.

**pseudopeople requires Python 3.11 or newer to run**

You can install pseudopeople from PyPI with pip:

  ``> pip install pseudopeople``

Datasets are returned as `Polars <https://pola.rs/>`_ DataFrames. To also be
able to convert them to pandas with ``.to_pandas()``, install the ``pandas`` extra:

  ``> pip install pseudopeople[pandas]``

For development, clone the repository and use `uv <https://docs.astral.sh/uv/>`_:

  ``> git clone https://github.com/ihmeuw/pseudopeople.git``

  ``> cd pseudopeople``

  ``> uv sync``

  ``> uv run pytest``

Documentation
======================
You can view documentation at https://pseudopeople.readthedocs.io/en/latest/

To receive updates about this software package `join the mailing list
here
<https://mailman11.u.washington.edu/mailman/listinfo/pseudopeople-users>`_.
