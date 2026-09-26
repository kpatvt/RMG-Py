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
Tests that generating reactions with symmetry compression (shadow reactions) gives exactly the same
results as generating every reaction.
"""

import itertools
import os

from rmgpy import settings
import rmgpy.data.kinetics.family as family_module
from rmgpy.data.base import ForbiddenStructures
from rmgpy.data.kinetics.common import (
    IdentityKey,
    ShadowReaction,
    ensure_independent_atom_ids,
    find_degenerate_reactions,
    reaction_identity_key,
)
from rmgpy.data.rmg import RMGDatabase
from rmgpy.species import Species

SPECIES_SETS = [
    ("CC(C)(C)C",),
    ("C1CCCCC1",),
    ("[CH2]CCC",),
    ("CC(C)(C)C", "[H]"),
    ("CC(C)(C)C", "[CH3]"),
    ("CC(C)C", "[O]O"),
    ("C=CC=C", "[H]"),
    ("C=CC=C", "[CH2]C=C"),
    ("[CH2]C=C", "[CH2]C=C"),
    ("[CH3]", "[CH3]"),
    ("C1CCCCC1", "[OH]"),
    ("CC(C)=C(C)C", "[CH3]"),
    ("c1ccccc1", "[H]"),
    ("[CH2]c1ccccc1", "CC"),
    ("CCCCCOO", "[O]O"),
    ("OO", "[OH]"),
]


def setup_module():
    """Load the testing database once for all tests in this module."""
    global database
    database = RMGDatabase()
    database.load(
        path=os.path.join(settings["test_data.directory"], "testing_database"),
        thermo_libraries=["primaryThermoLibrary"],
        reaction_libraries=[],
        kinetics_families=[
            "R_Recombination",
            "Disproportionation",
            "R_Addition_MultipleBond",
            "H_Abstraction",
            "intra_H_migration",
        ],
        testing=True,
        depository=False,
        solvation=False,
        surface=False,
    )
    for family in database.kinetics.families.values():
        family.forbidden = ForbiddenStructures()
    database.forbidden_structures = ForbiddenStructures()


def teardown_module():
    from rmgpy.data import rmg

    rmg.database = None
    family_module.SYMMETRY_COMPRESSION = True


def _describe(reactions):
    """Return a detailed description of the reactions, including atom IDs and the reverse reactions."""
    description = []
    # Atom IDs come from a global counter, so number them in order of appearance
    ids = {}
    for rxn in reactions:
        description.append((str(rxn), rxn.family, rxn.degeneracy, rxn.duplicate, sorted(rxn.template)))
        if getattr(rxn, "reverse", None) is not None:
            description.append(("reverse", rxn.reverse.degeneracy, sorted(rxn.reverse.template)))
        for spc in rxn.reactants + rxn.products:
            for mol in spc.molecule:
                description.append(tuple((ids.setdefault(atom.id, len(ids)), atom.symbol, atom.label)
                                         for atom in mol.atoms))
                description.append(mol.to_adjacency_list())
    return description


def _generate(smiles, compression):
    family_module.SYMMETRY_COMPRESSION = compression
    try:
        species = [Species().from_smiles(smi) for smi in smiles]
        return _describe(database.kinetics.generate_reactions_from_families(species))
    finally:
        family_module.SYMMETRY_COMPRESSION = True


def test_compressed_generation_is_identical():
    """Reactions generated with and without symmetry compression must be identical"""
    for smiles in SPECIES_SETS:
        assert _generate(smiles, False) == _generate(smiles, True), smiles


def test_shadow_reactions_are_generated():
    """Symmetric reactants give shadow reactions, which find_degenerate_reactions removes"""
    family = database.kinetics.families["H_Abstraction"]
    reactants = [Species().from_smiles("CC(C)(C)C"), Species().from_smiles("[H]")]
    ensure_independent_atom_ids(reactants)
    molecules = [reactants[0].molecule[0], reactants[1].molecule[0]]
    reactions = family.generate_reactions(molecules, compress_symmetric=True)
    shadows = [rxn for rxn in reactions if isinstance(rxn, ShadowReaction)]
    # The 12 hydrogen atoms of neopentane are equivalent: one reaction and 11 shadows
    assert len(reactions) == 12
    assert len(shadows) == 11
    degenerate = find_degenerate_reactions(reactions)
    assert len(degenerate) == 1
    assert not isinstance(degenerate[0], ShadowReaction)
    assert degenerate[0].degeneracy == 12

    uncompressed = family.generate_reactions(molecules, compress_symmetric=False)
    assert not any(isinstance(rxn, ShadowReaction) for rxn in uncompressed)
    assert len(uncompressed) == 12


def test_identity_key_matches_identity_check():
    """IdentityKey equality must agree with the identity check of find_degenerate_reactions"""
    for smiles in [("[CH2]C=C", "[CH2]C=C"), ("C=CC=C", "[H]"), ("CC(C)C", "[O]O"), ("c1ccccc1", "[H]")]:
        reactants = [Species().from_smiles(smi) for smi in smiles]
        if len(reactants) == 2 and reactants[0].is_isomorphic(reactants[1]):
            reactants[1] = reactants[1].copy(deep=True)
        ensure_independent_atom_ids(reactants)
        combos = list(itertools.product(reactants[0].molecule, reactants[1].molecule))
        for family in database.kinetics.families.values():
            reactions = []
            for combo in combos:
                reactions.extend(family.generate_reactions(list(combo)))
            for rxn in reactions:
                rxn.ensure_species()
            keys = [reaction_identity_key(rxn) for rxn in reactions]
            for (rxn1, key1), (rxn2, key2) in itertools.combinations(zip(reactions, keys), 2):
                assert isinstance(key1, IdentityKey) and isinstance(key2, IdentityKey)
                identical = rxn1.is_isomorphic(rxn2, check_identical=True, strict=False,
                                               check_template_rxn_products=True)
                assert (key1 == key2) == identical, (str(rxn1), str(rxn2))


def test_product_connectivity_filter():
    """The product pre-check must never reject a mapping whose products match, and should reject others"""
    from rmgpy.data.kinetics.family import _ProductConnectivityFilter
    from rmgpy.reaction import same_species_lists

    family = database.kinetics.families["H_Abstraction"]
    for smiles1, smiles2 in [("CCC", "[OH]"), ("CC(C)C", "[O]O"), ("C=CC", "[CH3]")]:
        reactants = [Species().from_smiles(smiles1), Species().from_smiles(smiles2)]
        ensure_independent_atom_ids(reactants)
        molecules = [reactants[0].molecule[0], reactants[1].molecule[0]]
        reactions = family.generate_reactions(molecules)
        assert len(reactions) > 1
        # Use the products of each reaction in turn as the requested products
        for target in reactions:
            products = [p.copy(deep=True) for p in target.products]
            product_filter = _ProductConnectivityFilter(family.forward_recipe, products)
            template = family.forward_template.reactants
            rejected = 0
            for map_a in family._match_reactant_to_template(molecules[0], template[0].item):
                for map_b in family._match_reactant_to_template(molecules[1], template[1].item):
                    for structures, maps in (([molecules[0], molecules[1]], [map_a, map_b]),):
                        may_match = product_filter.may_match(structures, maps)
                        try:
                            generated = family._generate_product_structures(structures, maps, True)
                        except Exception:
                            generated = None
                        if generated is not None and same_species_lists(products, generated, strict=False):
                            assert may_match
                        if not may_match:
                            rejected += 1
            assert rejected > 0
