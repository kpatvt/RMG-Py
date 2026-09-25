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
Unit tests for the rmgpy.rmg.database_cache module.
"""

import os

import rmgpy.data.rmg
import rmgpy.rmg.database_cache as database_cache
from rmgpy.rmg.main import RMG


class TestDatabaseCache:
    def _make_rmg(self, database_directory):
        rmg = RMG()
        rmg.database_directory = str(database_directory)
        rmg.thermo_libraries = ["primaryThermoLibrary"]
        rmg.transport_libraries = None
        rmg.reaction_libraries = []
        rmg.seed_mechanisms = []
        rmg.kinetics_families = "default"
        rmg.kinetics_depositories = ["training"]
        rmg.statmech_libraries = []
        rmg.adsorption_groups = "adsorptionPt111"
        rmg.kinetics_estimator = "rate rules"
        return rmg

    def test_cache_file(self, tmp_path, monkeypatch):
        database_directory = tmp_path / "database"
        database_directory.mkdir()
        (database_directory / "file.py").write_text("x = 1\n")
        rmg = self._make_rmg(database_directory)

        monkeypatch.delenv(database_cache.CACHE_ENVIRONMENT_VARIABLE, raising=False)
        assert database_cache.get_cache_file(rmg) is None

        monkeypatch.setenv(database_cache.CACHE_ENVIRONMENT_VARIABLE, str(tmp_path / "cache"))
        cache_file = database_cache.get_cache_file(rmg)
        assert os.path.dirname(cache_file) == str(tmp_path / "cache")
        assert database_cache.get_cache_file(rmg) == cache_file

        # Changing a setting or a database file changes the cache file
        rmg.thermo_libraries = ["primaryThermoLibrary", "BurkeH2O2"]
        assert database_cache.get_cache_file(rmg) != cache_file
        rmg.thermo_libraries = ["primaryThermoLibrary"]
        (database_directory / "file.py").write_text("x = 12\n")
        assert database_cache.get_cache_file(rmg) != cache_file

        # The cache is not used when the prepared database depends on QM or ML settings
        rmg.ml_estimator = True
        assert database_cache.get_cache_file(rmg) is None

    def test_save_and_load(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "cache" / "database.pickle")
        assert database_cache.load(cache_file) is None
        database_cache.save({"a": [1, 2.5]}, cache_file)
        assert os.listdir(tmp_path / "cache") == ["database.pickle"]
        monkeypatch.setattr(rmgpy.data.rmg, "database", None)
        database = database_cache.load(cache_file)
        assert database == {"a": [1, 2.5]}
        # The loaded database is registered as the global database
        assert rmgpy.data.rmg.database is database

        # An unreadable cache file is ignored
        with open(cache_file, "wb") as f:
            f.write(b"not a pickle")
        assert database_cache.load(cache_file) is None
