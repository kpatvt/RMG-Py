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
This module contains classes and functions that are used by multiple modules
in this subpackage.
"""
import itertools
import logging

from rmgpy.data.base import LogicNode
from rmgpy.exceptions import DatabaseError
from rmgpy.molecule import Group, Molecule
from rmgpy.molecule.fragment import Fragment
from rmgpy.reaction import Reaction, same_species_lists
from rmgpy.species import Species


################################################################################


def save_entry(f, entry):
    """
    Save an `entry` in the kinetics database by writing a string to
    the given file object `f`.
    """

    def sort_efficiencies(efficiencies0):
        efficiencies = {}
        for mol, eff in efficiencies0.items():
            if isinstance(mol, str):
                # already in SMILES string format
                smiles = mol
            else:
                smiles = mol.to_smiles()

            efficiencies[smiles] = eff
        keys = list(efficiencies.keys())
        keys.sort()
        return [(key, efficiencies[key]) for key in keys]

    f.write('entry(\n')
    f.write('    index = {0:d},\n'.format(entry.index))
    if entry.label != '':
        f.write('    label = "{0}",\n'.format(entry.label))

    # Entries for kinetic rules, libraries, training reactions
    # and depositories will have a Reaction object for its item
    if isinstance(entry.item, Reaction):
        # Write out additional data if depository or library
        # kinetic rules would have a Group object for its reactants instead of Species
        if isinstance(entry.item.reactants[0], Species):
            # Add degeneracy if the reaction is coming from a depository or kinetics library
            f.write('    degeneracy = {0:.1f},\n'.format(entry.item.degeneracy))
            if entry.item.duplicate:
                f.write('    duplicate = {0!r},\n'.format(entry.item.duplicate))
            if not entry.item.reversible:
                f.write('    reversible = {0!r},\n'.format(entry.item.reversible))
            if entry.item.allow_pdep_route:
                f.write('    allow_pdep_route = {0!r},\n'.format(entry.item.allow_pdep_route))
            if entry.item.elementary_high_p:
                f.write('    elementary_high_p = {0!r},\n'.format(entry.item.elementary_high_p))
            if entry.item.allow_max_rate_violation:
                f.write('    allow_max_rate_violation = {0!r},\n'.format(entry.item.allow_max_rate_violation))
    # Entries for groups with have a group or logicNode for its item
    elif isinstance(entry.item, Group):
        f.write('    group = \n')
        f.write('"""\n')
        f.write(entry.item.to_adjacency_list())
        f.write('""",\n')
    elif isinstance(entry.item, LogicNode):
        f.write('    group = "{0}",\n'.format(entry.item))
    else:
        raise DatabaseError("Encountered unexpected item of type {0} while "
                            "saving database.".format(entry.item.__class__))

    # Write kinetics
    if isinstance(entry.data, str):
        f.write('    kinetics = "{0}",\n'.format(entry.data))
    elif entry.data is not None:
        efficiencies = None
        if hasattr(entry.data, 'efficiencies'):
            efficiencies = entry.data.efficiencies
            entry.data.efficiencies = dict(sort_efficiencies(entry.data.efficiencies))
        kinetics = repr(entry.data)  # todo prettify currently does not support uncertainty attribute
        kinetics = '    kinetics = {0},\n'.format(kinetics.replace('\n', '\n    '))
        f.write(kinetics)
        if hasattr(entry.data, 'efficiencies'):
            entry.data.efficiencies = efficiencies
    else:
        f.write('    kinetics = None,\n')

    # Write reference
    if entry.reference is not None:
        reference = entry.reference.to_pretty_repr()
        lines = reference.splitlines()
        f.write('    reference = {0}\n'.format(lines[0]))
        for line in lines[1:-1]:
            f.write('    {0}\n'.format(line))
        f.write('    ),\n'.format(lines[0]))

    if entry.reference_type != "":
        f.write('    referenceType = "{0}",\n'.format(entry.reference_type))
    if entry.rank is not None:
        f.write('    rank = {0},\n'.format(entry.rank))

    if entry.short_desc.strip() != '':
        f.write(f'    shortDesc = """{entry.short_desc.strip()}""",\n')
    if entry.long_desc.strip() != '':
        f.write(f'    longDesc = \n"""\n{entry.long_desc.strip()}\n""",\n')

    # write metal attributes
    if entry.metal:
        f.write('    metal = "{0}",\n'.format(entry.metal))
    if entry.facet:
        f.write('    facet = "{0}",\n'.format(entry.facet))
    if entry.site:
        f.write('    site = "{0}",\n'.format(entry.site))

    f.write(')\n\n')


def ensure_species(input_list, resonance=False, keep_isomorphic=False):
    """
    The input list of :class:`Species` or :class:`Molecule` objects is modified
    in place to only have :class:`Species` objects. Returns None.
    """
    for index, item in enumerate(input_list):
        if isinstance(item, Molecule) or isinstance(item, Fragment):
            new_item = Species(molecule=[item])
        elif isinstance(item, Species):
            new_item = item
        else:
            raise TypeError('Only Molecule or Species objects can be handled.')
        if resonance:
            if not any([mol.reactive for mol in new_item.molecule]):
                # if generating a reaction containing a Molecule with a reactive=False flag (e.g., for degeneracy
                # calculations), that was now converted into a Species, first mark as reactive=True
                new_item.molecule[0].reactive = True
            new_item.generate_resonance_structures(keep_isomorphic=keep_isomorphic)
        input_list[index] = new_item


def generate_molecule_combos(input_species):
    """
    Generate combinations of molecules from the given species objects.
    """
    if len(input_species) == 1:
        combos = [(mol,) for mol in input_species[0].molecule]
    elif len(input_species) == 2:
        combos = itertools.product(input_species[0].molecule, input_species[1].molecule)
    elif len(input_species) == 3:
        combos = itertools.product(input_species[0].molecule, input_species[1].molecule, input_species[2].molecule)
    else:
        raise ValueError('Reaction generation can be done for 1, 2, or 3 species, not {0}.'.format(len(input_species)))

    return combos


def ensure_independent_atom_ids(input_species, resonance=True):
    """
    Given a list or tuple of :class:`Species` or :class:`Molecule` objects,
    ensure that atom ids are independent.
    The `resonance` argument can be set to False to not generate
    resonance structures.

    Modifies the list in place (replacing :class:`Molecule` with :class:`Species`).
    Returns None.
    """
    ensure_species(input_species)  # do not generate resonance structures since we do so below

    # Inline check that all atom IDs across all species' first molecule are unique.
    # Building a list then converting to set is faster than incremental set.add()
    # because set(list) has a vectorized C path.
    ids = []
    for spcs in input_species:
        atoms = spcs.molecule[0].atoms
        ids.extend([atom.id for atom in atoms])

    if len(set(ids)) != len(ids):
        # Collision: reassign IDs and remake resonance structures
        for species in input_species:
            reactive_mols = []
            unreactive_mols = []
            for m in species.molecule:
                (reactive_mols if m.reactive else unreactive_mols).append(m)
            mol = reactive_mols[0]  # Choose first reactive molecule
            mol.assign_atom_ids()
            species.molecule = [mol]
            # Remake resonance structures with new labels
            if resonance:
                species.generate_resonance_structures(keep_isomorphic=True)
            if unreactive_mols:
                species.molecule.extend(unreactive_mols)
    elif resonance:
        # IDs are already independent, generate resonance structures if needed
        for species in input_species:
            species.generate_resonance_structures(keep_isomorphic=True)

def _independent_copy(reactant):
    """
    Return a copy of the `reactant` (a :class:`Species` or :class:`Molecule`) whose molecules can be
    manipulated independently of those of `reactant`, for reacting a species with itself.

    Unlike ``Species.copy(deep=True)``, this does not copy the thermo, conformer, transport and
    energy transfer data (which are shared with `reactant`): only the molecules of the copy are
    used when generating reactions, and copying the data is expensive.
    """
    if not isinstance(reactant, Species):
        return reactant.copy(deep=True)
    other = Species.__new__(Species)
    other.index = reactant.index
    other.label = reactant.label
    other.thermo = reactant.thermo
    other.molecule = [molecule.copy(deep=True) for molecule in reactant.molecule]
    other.conformer = reactant.conformer
    other.transport_data = reactant.transport_data
    other.molecular_weight = reactant.molecular_weight
    other.energy_transfer_model = reactant.energy_transfer_model
    other.reactive = reactant.reactive
    other.props = dict(reactant.props)
    return other


def check_for_same_reactants(reactants):
    """
    Given a list reactants, check if the reactants are the same.
    If they refer to the same memory address, then make a deep copy so they can be manipulated independently.
    
    Returns a tuple containing the modified reactants list, and an integer containing the number of identical reactants in the reactants list. 
    
    """

    same_reactants = 0
    if len(reactants) == 2:
        if reactants[0] is reactants[1]:
            reactants[1] = _independent_copy(reactants[1])
            same_reactants = 2
        elif reactants[0].is_isomorphic(reactants[1]):
            same_reactants = 2
    elif len(reactants) == 3:
        same_01 = reactants[0] is reactants[1]
        same_02 = reactants[0] is reactants[2]
        if same_01 and same_02:
            same_reactants = 3
            reactants[1] = _independent_copy(reactants[1])
            reactants[2] = _independent_copy(reactants[2])
        elif same_01:
            same_reactants = 2
            reactants[1] = _independent_copy(reactants[1])
        elif same_02:
            same_reactants = 2
            reactants[2] = _independent_copy(reactants[2])
        elif reactants[1] is reactants[2]:
            same_reactants = 2
            reactants[2] = _independent_copy(reactants[2])
        else:
            same_01 = reactants[0].is_isomorphic(reactants[1])
            same_02 = reactants[0].is_isomorphic(reactants[2])
            if same_01 and same_02:
                same_reactants = 3
            elif same_01 or same_02:
                same_reactants = 2
            elif reactants[1].is_isomorphic(reactants[2]):
                same_reactants = 2
    elif len(reactants) > 3:
        raise ValueError('Cannot check for duplicate reactants if provided number of reactants is greater than 3. ' 
                         'Got: {} reactants'.format(len(reactants))) 
        
    return reactants, same_reactants

def _template_product_formulas(rxn):
    """
    Return the sorted formulas (fingerprints) of the species that are compared when checking
    `rxn` for isomorphism with ``check_template_rxn_products=True``, or ``None`` if any of them
    is unavailable. Two such reactions can only be isomorphic if these are equal.
    """
    is_forward = getattr(rxn, 'is_forward', None)
    if is_forward is None:
        return None
    species = rxn.products if is_forward else rxn.reactants
    formulas = []
    for spc in species:
        fingerprint = spc.fingerprint
        if not fingerprint:
            return None
        formulas.append(fingerprint)
    return tuple(sorted(formulas))


def _identity_signature(rxn):
    """
    Return, for each of the species compared when checking `rxn` for identity with
    ``check_template_rxn_products=True``, the set of the atom ID sets of its molecules, or ``None``
    if this is not possible.
    """
    is_forward = getattr(rxn, 'is_forward', None)
    if is_forward is None:
        return None
    signature = []
    for spc in (rxn.products if is_forward else rxn.reactants):
        molecules = spc.molecule if isinstance(spc, Species) else [spc]
        signature.append(frozenset(frozenset(atom.id for atom in mol.atoms) for mol in molecules))
    return signature


def _could_be_identical(signatures, rxn1, rxn2):
    """
    Return ``False`` if `rxn1` and `rxn2` can certainly not be identical (as checked by
    ``Reaction.is_isomorphic(check_identical=True, check_template_rxn_products=True)``).

    Two molecules can only be identical if they have the same set of atom IDs, so two species can
    only be identical if one of their molecules' atom ID sets is the same, and the species lists
    can only be identical if they can be paired up such that this is the case for each pair.
    The signatures are cached in the dict `signatures` by reaction id.
    """
    for rxn in (rxn1, rxn2):
        if id(rxn) not in signatures:
            signatures[id(rxn)] = (rxn, _identity_signature(rxn))
    signature1 = signatures[id(rxn1)][1]
    signature2 = signatures[id(rxn2)][1]
    if signature1 is None or signature2 is None:
        return True
    if len(signature1) != len(signature2):
        return False
    for order in itertools.permutations(range(len(signature2))):
        if all(signature1[i] & signature2[j] for i, j in enumerate(order)):
            return True
    return False


class ShadowReaction(object):
    """
    A placeholder for a generated reaction that is related to an earlier generated reaction of the
    same reaction family, its `representative`, by a symmetry of the reactants: the reactant atoms
    matched to the template are mapped onto those of the representative by an automorphism of the
    reactants. The reaction is therefore isomorphic to its representative (with the same template),
    and all of the family's checks treat both alike, so its products are not generated. It only
    differs from its representative by which atoms react, which is what the degeneracy counts, and
    :func:`find_degenerate_reactions` accounts for it using its `identity_key` (see
    :class:`IdentityKey`). A shadow reaction never appears in the output of
    :func:`find_degenerate_reactions`; if it would, the actual reaction is generated by `materialize`.
    """

    __slots__ = ('representative', 'identity_key', 'is_forward', 'family', 'template', 'degeneracy', 'duplicate',
                 'reversible', '_materialize')

    def __init__(self, representative, identity_key, materialize):
        self.representative = representative
        self.identity_key = identity_key
        self.is_forward = representative.is_forward
        self.family = representative.family
        self.template = None
        self.degeneracy = 1
        self.duplicate = False
        self.reversible = representative.reversible
        self._materialize = materialize

    def materialize(self):
        """
        Generate and return the actual reaction.
        """
        return self._materialize()


def _atom_id_graph(molecules):
    """
    Return a frozenset of the atom IDs with their elements (symbol and isotope) and a frozenset of
    the bonds (as sorted pairs of atom IDs) of the given molecules, or ``None`` if the atom IDs are
    not unique.
    """
    atoms = []
    edges = []
    for molecule in molecules:
        for atom in molecule.atoms:
            atom_id = atom.id
            atoms.append((atom_id, atom.element.symbol, atom.element.isotope))
            for neighbor in atom.edges:
                if atom_id < neighbor.id:
                    edges.append((atom_id, neighbor.id))
    atom_set = frozenset(atoms)
    if len(set([atom[0] for atom in atoms])) != len(atoms):
        return None
    return atom_set, frozenset(edges)


def _component_count(atoms, edges):
    """
    Return the number of connected components of the graph of the given atoms and bonds.
    """
    parent = {atom[0]: atom[0] for atom in atoms}

    def find(atom_id):
        while parent[atom_id] != atom_id:
            parent[atom_id] = parent[parent[atom_id]]
            atom_id = parent[atom_id]
        return atom_id

    count = len(parent)
    for id1, id2 in edges:
        root1, root2 = find(id1), find(id2)
        if root1 != root2:
            parent[root1] = root2
            count -= 1
    return count


class IdentityKey(object):
    """
    A key such that two reactions are identical, as checked by ``Reaction.is_isomorphic(
    check_identical=True, strict=False, check_template_rxn_products=True)``, if and only if their
    keys are equal (see :func:`reaction_identity_key`).

    With ``strict=False``, two molecules are identical if they have the same atom IDs, the same
    element for each ID, and bonds between the same pairs of IDs. The generated species of a
    reaction have disjoint atom IDs, so two reactions are identical if the generated species have
    the same atoms and the same bonds between them, given that the generated species are the
    connected components of their bond graph (which the keys are only made for). The bonds are
    stored as those of the template reactants (which are usually shared by the reactions compared)
    and the bonds broken and formed.
    """

    __slots__ = ('atoms', 'reactant_edges', 'broken', 'formed')

    def __init__(self, atoms, reactant_edges, broken, formed):
        self.atoms = atoms
        self.reactant_edges = reactant_edges
        self.broken = broken
        self.formed = formed

    def edges(self):
        """
        Return the bonds of the generated species.
        """
        return (self.reactant_edges - self.broken) | self.formed

    def __eq__(self, other):
        if not isinstance(other, IdentityKey):
            return NotImplemented
        if self.atoms != other.atoms:
            return False
        if self.reactant_edges is other.reactant_edges or self.reactant_edges == other.reactant_edges:
            return self.broken == other.broken and self.formed == other.formed
        return self.edges() == other.edges()

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = None


def _single_molecules(species_list):
    """
    Return the molecules of the given species (or molecules) if each has a single molecule,
    else ``None``.
    """
    molecules = []
    for spc in species_list:
        if isinstance(spc, Species):
            if len(spc.molecule) != 1:
                return None
            molecule = spc.molecule[0]
        else:
            molecule = spc
        if type(molecule) is not Molecule:
            return None
        molecules.append(molecule)
    return molecules


def reaction_identity_key(rxn, graph_cache=None):
    """
    Return the :class:`IdentityKey` of `rxn`, or ``None`` if one cannot be made. The atom ID graphs
    of the template reactants can be cached in the dict `graph_cache` by the ids of the molecules.
    """
    if isinstance(rxn, ShadowReaction):
        return rxn.identity_key
    generated = _single_molecules(rxn.products if rxn.is_forward else rxn.reactants)
    template_reactants = _single_molecules(rxn.reactants if rxn.is_forward else rxn.products)
    if generated is None or template_reactants is None:
        return None
    cache_key = tuple([id(molecule) for molecule in template_reactants])
    if graph_cache is not None and cache_key in graph_cache:
        reactant_graph = graph_cache[cache_key][1]
    else:
        reactant_graph = _atom_id_graph(template_reactants)
        if graph_cache is not None:
            graph_cache[cache_key] = (template_reactants, reactant_graph)
    product_graph = _atom_id_graph(generated)
    if reactant_graph is None or product_graph is None or reactant_graph[0] != product_graph[0]:
        return None
    atoms, reactant_edges = reactant_graph
    product_edges = product_graph[1]
    if _component_count(atoms, product_edges) != len(generated):
        return None
    return IdentityKey(atoms, reactant_edges, reactant_edges - product_edges, product_edges - reactant_edges)


def representative_bond_changes(reactant_structures, product_structures, maps):
    """
    For a reaction generated from `reactant_structures` by the template `maps`, return the atom ID
    graph of the reactants and the bonds broken and formed, as pairs of the template atoms they
    connect, or ``None`` if the reaction cannot be used as the representative of shadow reactions.
    """
    reactant_graph = _atom_id_graph(reactant_structures)
    product_graph = _atom_id_graph(product_structures)
    if reactant_graph is None or product_graph is None or reactant_graph[0] != product_graph[0]:
        return None
    atoms, reactant_edges = reactant_graph
    product_edges = product_graph[1]
    if _component_count(atoms, product_edges) != len(product_structures):
        return None
    group_atoms = {}
    for mapping in maps:
        for atom, group_atom in mapping.items():
            group_atoms[atom.id] = group_atom
    try:
        broken = [(group_atoms[id1], group_atoms[id2]) for id1, id2 in reactant_edges - product_edges]
        formed = [(group_atoms[id1], group_atoms[id2]) for id1, id2 in product_edges - reactant_edges]
    except KeyError:
        return None
    return atoms, reactant_edges, broken, formed


def shadow_identity_key(bond_changes, maps):
    """
    Return the :class:`IdentityKey` of the reaction generated by the template `maps` from the same
    reactant structures as a representative reaction with the given `bond_changes` (see
    :func:`representative_bond_changes`), when the maps are related by an automorphism of the
    reactants: the bonds formed and broken are those between the same template atoms. Returns
    ``None`` if this is not possible.
    """
    atoms, reactant_edges, broken_group, formed_group = bond_changes
    ids = {}
    for mapping in maps:
        for atom, group_atom in mapping.items():
            ids[group_atom] = atom.id
    broken = []
    formed = []
    try:
        for group_atom1, group_atom2 in broken_group:
            id1, id2 = ids[group_atom1], ids[group_atom2]
            edge = (id1, id2) if id1 < id2 else (id2, id1)
            if edge not in reactant_edges:
                return None
            broken.append(edge)
        for group_atom1, group_atom2 in formed_group:
            id1, id2 = ids[group_atom1], ids[group_atom2]
            edge = (id1, id2) if id1 < id2 else (id2, id1)
            if edge in reactant_edges:
                return None
            formed.append(edge)
    except KeyError:
        return None
    return IdentityKey(atoms, reactant_edges, frozenset(broken), frozenset(formed))


def find_degenerate_reactions(rxn_list, same_reactants=None, template=None, kinetics_database=None,
                              kinetics_family=None, save_order=False, resonance=True):
    """
    Given a list of Reaction objects, this method combines degenerate
    reactions and increments the reaction degeneracy value. For multiple
    transition states, this method keeps them as duplicate reactions.

    If a template is specified, then the reaction list will be filtered
    to leave only reactions which match the specified template, then the
    degeneracy will be calculated as usual.

    A KineticsDatabase or KineticsFamily instance can also be provided to
    calculate the degeneracy for reactions generated in the reverse direction.
    If not provided, then it will be retrieved from the global database.

    This algorithm used to exist in family._generate_reactions, but was moved
    here so it could operate across reaction families.

    This method returns an updated list with degenerate reactions removed.

    Args:
        rxn_list (list):                                reactions to be analyzed
        same_reactants (bool, optional):                indicate whether the reactants are identical
        template (list, optional):                      specify a specific template to filter by
        kinetics_database (KineticsDatabase, optional): provide a KineticsDatabase instance for calculating degeneracy
        kinetics_family (KineticsFamily, optional):     provide a KineticsFamily instance for calculating degeneracy
        save_order (bool, optional):                    reset atom order after performing atom isomorphism
        resonance (bool, optional):                     whether to consider resonance when computing degeneracy 

    Returns:
        Reaction list with degenerate reactions combined with proper degeneracy values
    """
    # If a specific reaction template is requested, filter by that template
    if template is not None:
        selected_rxns = []
        template = frozenset(template)
        for rxn in rxn_list:
            if template == frozenset(rxn.template):
                selected_rxns.append(rxn)
        if not selected_rxns:
            # Only log a warning here. If a non-empty output is expected, then the caller should raise an exception
            logging.warning('No reactions matched the specified template, {0}'.format(template))
            return []
    else:
        selected_rxns = rxn_list

    # We want to sort all the reactions into sublists composed of isomorphic reactions
    # with degenerate transition states
    sorted_rxns = []
    # The product formulas of the reactions in each sublist. Reactions are compared by the
    # isomorphism of their products, which requires matching formulas, so sublists with
    # different product formulas can be skipped without running the isomorphism checks.
    sorted_keys = []
    # Atom ID signatures of the reactions, see _could_be_identical()
    identity_signatures = {}
    # State used for shadow reactions (see ShadowReaction): the results of the isomorphism checks of
    # reactions against the first reaction of each sublist, by (reaction id, sublist id), the sublist
    # each reaction was placed in (or found identical to a reaction of), identity keys, and copies of
    # the products of representatives
    shadow_state = {'isomorphic': {}, 'home': {}, 'keys': {}, 'surrogates': {}, 'graphs': {}, 'keep': []}
    for rxn0 in selected_rxns:
        if isinstance(rxn0, ShadowReaction):
            if _place_shadow_reaction(rxn0, sorted_rxns, sorted_keys, shadow_state, save_order):
                continue
            # The shadow reaction cannot be placed without the actual reaction
            rxn0 = rxn0.materialize()
        rxn0.ensure_species(save_order=save_order)
        key0 = _template_product_formulas(rxn0)
        if len(sorted_rxns) == 0:
            # This is the first reaction, so create a new sublist
            sorted_rxns.append([rxn0])
            sorted_keys.append(key0)
            shadow_state['home'][id(rxn0)] = sorted_rxns[-1]
        else:
            # Loop through each sublist, which represents a unique reaction
            for sub_list, key in zip(sorted_rxns, sorted_keys):
                if key0 is not None and key is not None and key0 != key:
                    # The products cannot be isomorphic, so this is not the right sublist
                    continue
                # Try to determine if the current rxn0 is identical or isomorphic to any reactions in the sublist
                isomorphic = False
                identical = False
                same_template = True
                for index, rxn in enumerate(sub_list):
                    if index == 0:
                        isomorphic = rxn0.is_isomorphic(rxn, check_identical=False, strict=False,
                                                        check_template_rxn_products=True, save_order=save_order)
                        shadow_state['isomorphic'][(id(rxn0), id(sub_list))] = isomorphic
                    # else: all reactions in a sublist are isomorphic to its first reaction (and isomorphism
                    # is transitive), so rxn0 is isomorphic to the others if it is isomorphic to the first
                    if isomorphic:
                        if isinstance(rxn, ShadowReaction):
                            identity_key0 = _identity_key(rxn0, shadow_state)
                            if identity_key0 is None:
                                # Compare with the actual reaction instead
                                rxn = sub_list[index] = rxn.materialize()
                                rxn.ensure_species(save_order=save_order)
                        if isinstance(rxn, ShadowReaction):
                            identical = identity_key0 == rxn.identity_key
                        else:
                            identical = (_could_be_identical(identity_signatures, rxn0, rxn) and
                                         rxn0.is_isomorphic(rxn, check_identical=True, strict=False,
                                                            check_template_rxn_products=True, save_order=save_order))
                        if identical:
                            # An exact copy of rxn0 is already in our list, so we can move on
                            break
                        same_template = frozenset(rxn.template) == frozenset(rxn0.template)
                    else:
                        # This sublist contains a different product
                        break

                # Process the reaction depending on the results of the comparisons
                if identical:
                    # This reaction does not contribute to degeneracy
                    shadow_state['home'][id(rxn0)] = sub_list
                    break
                elif isomorphic:
                    if same_template:
                        # We found the right sublist, and there is no identical reaction
                        # We should add rxn0 to the sublist as a degenerate rxn, and move on to the next rxn
                        sub_list.append(rxn0)
                        shadow_state['home'][id(rxn0)] = sub_list
                        break
                    else:
                        # We found an isomorphic sublist, but the reaction templates are different
                        # We need to mark this as a duplicate and continue searching the remaining sublists
                        rxn0.duplicate = True
                        sub_list[0].duplicate = True
                        continue
                else:
                    # This is not an isomorphic sublist, so we need to continue searching the remaining sublists
                    # Note: This else statement is not technically necessary but is included for clarity
                    continue
            else:
                # We did not break, which means that there was no isomorphic sublist, so create a new one
                sorted_rxns.append([rxn0])
                sorted_keys.append(key0)
                shadow_state['home'][id(rxn0)] = sorted_rxns[-1]
        # Keep the reactions whose ids are used as keys alive
        shadow_state['keep'].append(rxn0)

    rxn_list = []
    for sub_list in sorted_rxns:
        # Collapse our sorted reaction list by taking one reaction from each sublist
        rxn = sub_list[0]
        # The degeneracy of each reaction is the number of reactions that were in the sublist
        rxn.degeneracy = sum([reaction0.degeneracy for reaction0 in sub_list])
        rxn_list.append(rxn)

    for rxn in rxn_list:
        if rxn.is_forward:
            reduce_same_reactant_degeneracy(rxn, same_reactants)
        else:
            # fix the degeneracy of (not ownReverse) reactions found in the backwards direction
            try:
                family = kinetics_family or kinetics_database.families[rxn.family]
            except AttributeError:
                from rmgpy.data.rmg import get_db
                family = get_db('kinetics').families[rxn.family]
            if not family.own_reverse:
                rxn.degeneracy = family.calculate_degeneracy(rxn, resonance=resonance)

    return rxn_list


def _identity_key(rxn, shadow_state):
    """
    Return the identity key of `rxn` (see :func:`reaction_identity_key`), cached in `shadow_state`.
    """
    keys = shadow_state['keys']
    if id(rxn) not in keys:
        keys[id(rxn)] = (rxn, reaction_identity_key(rxn, shadow_state['graphs']))
    return keys[id(rxn)][1]


def _place_shadow_reaction(shadow, sorted_rxns, sorted_keys, shadow_state, save_order):
    """
    Place the `shadow` reaction into the sublists of :func:`find_degenerate_reactions` exactly as the
    actual reaction would be placed, and return ``True``, or return ``False`` if this requires the
    actual reaction (i.e. if it would start a new sublist, which does not normally happen, or its
    identity cannot be compared by key).

    The actual reaction is isomorphic to the representative of the shadow reaction (with the same
    template), so the representative's isomorphism check results are used. The identity checks use
    the identity keys.
    """
    representative = shadow.representative
    key0 = _template_product_formulas(representative)
    isomorphic_results = shadow_state['isomorphic']
    home = shadow_state['home'].get(id(representative))
    template = frozenset(shadow.template)
    for sub_list, key in zip(sorted_rxns, sorted_keys):
        if key0 is not None and key is not None and key0 != key:
            continue
        isomorphic = False
        identical = False
        same_template = True
        for index, rxn in enumerate(sub_list):
            if index == 0:
                if sub_list is home:
                    isomorphic = True
                else:
                    isomorphic = isomorphic_results.get((id(representative), id(sub_list)))
                    if isomorphic is None:
                        isomorphic = _shadow_is_isomorphic(representative, sub_list[0], shadow_state, save_order)
            if isomorphic:
                key = _identity_key(rxn, shadow_state)
                if key is None:
                    return False
                identical = key == shadow.identity_key
                if identical:
                    break
                same_template = frozenset(rxn.template) == template
            else:
                break
        if identical:
            return True
        elif isomorphic:
            if same_template:
                sub_list.append(shadow)
                return True
            else:
                shadow.duplicate = True
                sub_list[0].duplicate = True
                continue
    return False


def _shadow_is_isomorphic(representative, rxn, shadow_state, save_order):
    """
    Check a shadow reaction of `representative` for isomorphism with `rxn` as
    :func:`find_degenerate_reactions` does, using a copy of the representative's products (so that
    the check does not change the representative's products, e.g. their atom order).
    """
    surrogates = shadow_state['surrogates']
    if id(representative) not in surrogates:
        species = representative.products if representative.is_forward else representative.reactants
        surrogates[id(representative)] = (representative,
                                          [Species(molecule=[spc.molecule[0].copy(deep=True)]) for spc in species])
    species1 = surrogates[id(representative)][1]
    species2 = rxn.products if rxn.is_forward else rxn.reactants
    return same_species_lists(species1, species2, check_identical=False, only_check_label=False,
                              generate_initial_map=False, strict=False, save_order=save_order)


def reduce_same_reactant_degeneracy(reaction, same_reactants=None):
    """
    This method reduces the degeneracy of reactions with identical reactants,
    since translational component of the transition states are already taken
    into account (so swapping the same reactant is not valid)

    same_reactants can be None or an integer. If it is None, then isomorphism
    checks will be done to determine if the reactions are the same. If it is an
    integer, that integer denotes the number of reactants that are isomorphic.

    This comes from work by Bishop and Laidler in 1965
    """
    if not (same_reactants == 0 or same_reactants == 1):
        if len(reaction.reactants) == 2:
            if ((reaction.is_forward and same_reactants == 2) or
                    reaction.reactants[0].is_isomorphic(reaction.reactants[1])):
                reaction.degeneracy *= 0.5
                logging.debug(
                    'Degeneracy of reaction {} was decreased by 50% to {} since the reactants are identical'.format(
                        reaction, reaction.degeneracy)
                )
        elif len(reaction.reactants) == 3:
            if reaction.is_forward:
                if same_reactants == 3:
                    reaction.degeneracy /= 6.0
                    logging.debug(
                        'Degeneracy of reaction {} was divided by 6 to give {} since all of the reactants '
                        'are identical'.format(reaction, reaction.degeneracy)
                    )
                elif same_reactants == 2:
                    reaction.degeneracy *= 0.5
                    logging.debug(
                        'Degeneracy of reaction {} was decreased by 50% to {} since two of the reactants '
                        'are identical'.format(reaction, reaction.degeneracy)
                    )
            else:
                same_01 = reaction.reactants[0].is_isomorphic(reaction.reactants[1])
                same_02 = reaction.reactants[0].is_isomorphic(reaction.reactants[2])
                if same_01 and same_02:
                    reaction.degeneracy /= 6.0
                    logging.debug(
                        'Degeneracy of reaction {} was divided by 6 to give {} since all of the reactants '
                        'are identical'.format(reaction, reaction.degeneracy)
                    )
                elif same_01 or same_02:
                    reaction.degeneracy *= 0.5
                    logging.debug(
                        'Degeneracy of reaction {} was decreased by 50% to {} since two of the reactants '
                        'are identical'.format(reaction, reaction.degeneracy)
                    )
                elif reaction.reactants[1].is_isomorphic(reaction.reactants[2]):
                    reaction.degeneracy *= 0.5
                    logging.debug(
                        'Degeneracy of reaction {} was decreased by 50% to {} since two of the reactants '
                        'are identical'.format(reaction, reaction.degeneracy)
                    )
