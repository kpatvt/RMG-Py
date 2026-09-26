#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2026 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################


"""
An optional on-disk cache of the fully prepared RMG database.

Loading the RMG database and preparing it for a job (e.g. adding the rate rules from the training
reactions and averaging them) takes about 20-30 s per job, while reading a pickled copy of the
prepared database takes a few seconds. When the environment variable ``RMG_DATABASE_CACHE`` is set
to a directory, the prepared database is saved there after it is loaded, and later jobs with the
same settings load it from there instead.

A cached database is only used if everything it depends on is unchanged: the database files (their
paths, sizes and modification times), the RMG-Py code (likewise), the Python version, and all job
settings that affect the prepared database. Otherwise it is (re)built and saved as usual. The cache
is not used for jobs that estimate thermo with quantum mechanics or machine learning (which the
training reactions' thermo depends on) or that write the kinetics datastore.
"""

import gc
import hashlib
import logging
import os
import pickle
import sys

import rmgpy
import rmgpy.data.rmg

CACHE_ENVIRONMENT_VARIABLE = 'RMG_DATABASE_CACHE'
_CACHE_FORMAT_VERSION = 1


def _fingerprint_tree(root, extensions=None):
    """
    Return a hash of the relative paths, sizes and modification times of the files below `root`
    (optionally only those with the given `extensions`).
    """
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        for filename in sorted(filenames):
            if extensions is not None and not filename.endswith(extensions):
                continue
            path = os.path.join(dirpath, filename)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            digest.update('{0}\0{1}\0{2}\n'.format(os.path.relpath(path, root), stat.st_size,
                                                  stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _settings_description(rmg):
    """
    Return a description of all settings of the RMG job `rmg` that affect the prepared database.
    """
    forbidden = [(entry.label, entry.item.to_adjacency_list() if hasattr(entry.item, 'to_adjacency_list')
                  else repr(entry.item)) for entry in rmg.forbidden_structures]
    settings = [
        ('database_directory', os.path.abspath(rmg.database_directory)),
        ('thermo_libraries', rmg.thermo_libraries),
        ('transport_libraries', rmg.transport_libraries),
        ('reaction_libraries', [library for library, option in rmg.reaction_libraries]),
        ('seed_mechanisms', rmg.seed_mechanisms),
        ('kinetics_families', rmg.kinetics_families),
        ('kinetics_depositories', rmg.kinetics_depositories),
        ('statmech_libraries', rmg.statmech_libraries),
        ('adsorption_groups', rmg.adsorption_groups),
        ('trimolecular_product_reversible', rmg.trimolecular_product_reversible),
        ('binding_energies', repr(rmg.binding_energies)),
        ('forbidden_structures', forbidden),
        ('kinetics_estimator', rmg.kinetics_estimator),
        ('verbose_comments', rmg.verbose_comments),
        ('solvent', rmg.solvent),
    ]
    return repr(settings)


def get_cache_file(rmg):
    """
    Return the path of the cached prepared database for the RMG job `rmg`, or ``None`` if the
    cache is not enabled or cannot be used for this job.
    """
    cache_directory = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    if not cache_directory:
        return None
    if rmg.quantum_mechanics or rmg.ml_estimator or rmg.kinetics_datastore:
        logging.info('Not using the database cache, because the thermo of the training reactions depends '
                     'on quantum mechanics or machine learning settings, or the kinetics datastore is written.')
        return None
    digest = hashlib.sha256()
    for part in (
            str(_CACHE_FORMAT_VERSION),
            sys.version,
            str(pickle.HIGHEST_PROTOCOL),
            os.path.abspath(os.path.dirname(rmgpy.__file__)),
            _fingerprint_tree(os.path.dirname(rmgpy.__file__), extensions=('.py', '.so', '.pyd')),
            _fingerprint_tree(rmg.database_directory),
            _settings_description(rmg),
    ):
        digest.update(part.encode())
        digest.update(b'\0')
    return os.path.join(os.path.abspath(cache_directory), 'rmg_database_{0}.pickle'.format(digest.hexdigest()))


def load(cache_file):
    """
    Return the prepared database stored in `cache_file`, or ``None`` if there is none (or it
    cannot be read). The database is registered as the global RMG database.
    """
    if not os.path.isfile(cache_file):
        return None
    # Unpickling creates millions of objects, which would trigger many (futile) garbage collections
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with open(cache_file, 'rb') as f:
            database = pickle.load(f)
    except Exception as e:
        logging.warning('Could not load the cached database from {0} ({1}: {2}); loading the database '
                        'from its files instead.'.format(cache_file, type(e).__name__, e))
        return None
    finally:
        if gc_was_enabled:
            gc.enable()
    # Unpickling does not run RMGDatabase.__init__(), which registers the database globally
    rmgpy.data.rmg.database = database
    logging.info('Loaded the prepared database from the cache file {0}'.format(cache_file))
    return database


def save(database, cache_file):
    """
    Save the prepared `database` to `cache_file`.
    """
    temporary_file = '{0}.{1}.tmp'.format(cache_file, os.getpid())
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(temporary_file, 'wb') as f:
            pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_file, cache_file)
        logging.info('Saved the prepared database to the cache file {0}'.format(cache_file))
    except Exception as e:
        logging.warning('Could not save the database to the cache file {0} ({1}: {2})'.format(
            cache_file, type(e).__name__, e))
        try:
            os.remove(temporary_file)
        except OSError:
            pass
