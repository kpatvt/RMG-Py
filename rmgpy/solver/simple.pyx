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

# The loops of the residual and Jacobian are vectorized at -O3 (which does not change floating point results)
# distutils: extra_compile_args = -O3

"""
Contains the :class:`SimpleReactor` class, providing a reaction system
consisting of a homogeneous, isothermal, isobaric batch reactor.
"""

import itertools
import logging

cimport cython
import numpy as np
cimport numpy as np

import rmgpy.constants as constants
cimport rmgpy.constants as constants
from rmgpy.quantity import Quantity
from rmgpy.quantity cimport ScalarQuantity, ArrayQuantity
from rmgpy.solver.base cimport ReactionSystem
from rmgpy.kinetics.model cimport KineticsModel
from rmgpy.kinetics.falloff cimport ThirdBody, Lindemann, Troe


cdef double _pairwise_sum(double * a, Py_ssize_t n) noexcept nogil:
    """
    Return the sum of the `n` values starting at `a`, rounded exactly as ``np.sum()`` rounds it (numpy's
    pairwise summation: blocks of up to 128 values are summed with eight partial sums, and longer ranges are
    split in two), so that replacing ``np.sum()`` by this function does not change any result.
    """
    cdef Py_ssize_t i, n2
    cdef double res, r0, r1, r2, r3, r4, r5, r6, r7
    if n < 8:
        res = 0.
        for i in range(n):
            res += a[i]
        return res
    elif n <= 128:
        r0 = a[0]
        r1 = a[1]
        r2 = a[2]
        r3 = a[3]
        r4 = a[4]
        r5 = a[5]
        r6 = a[6]
        r7 = a[7]
        i = 8
        while i < n - (n % 8):
            r0 += a[i]
            r1 += a[i + 1]
            r2 += a[i + 2]
            r3 += a[i + 3]
            r4 += a[i + 4]
            r5 += a[i + 5]
            r6 += a[i + 6]
            r7 += a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i]
            i += 1
        return res
    else:
        n2 = n // 2
        n2 -= n2 % 8
        return _pairwise_sum(a, n2) + _pairwise_sum(a + n2, n - n2)


cdef inline np.ndarray _zeros(Py_ssize_t n):
    """
    Return a new array of `n` zeros (like ``np.zeros(n)``, but without the Python call overhead).
    """
    cdef np.npy_intp dims[1]
    dims[0] = n
    return np.PyArray_ZEROS(1, dims, np.NPY_FLOAT64, 0)


cdef double _empty_data[1]
cdef long _empty_indices[1]


cdef double * _float_data(np.ndarray a, Py_ssize_t n) except NULL:
    """
    Return a pointer to the data of the array of floats `a`, after checking that it is a contiguous array of
    at least `n` values.
    """
    if np.PyArray_TYPE(a) != np.NPY_FLOAT64 or not np.PyArray_IS_C_CONTIGUOUS(a):
        raise ValueError('Expected a contiguous array of floats')
    if np.PyArray_SIZE(a) < n:
        raise IndexError('Expected an array of at least {0} values, got {1}'.format(n, np.PyArray_SIZE(a)))
    if np.PyArray_SIZE(a) == 0:
        return _empty_data
    return <double *> np.PyArray_DATA(a)


cdef long * _int_data(np.ndarray a, Py_ssize_t n) except NULL:
    """
    Return a pointer to the data of the array of integers (``np.int_``) `a`, after checking that it is a contiguous
    array of at least `n` values. For a two-dimensional array of reactant or product indices, `n` must be three
    times the number of rows used, and the array must have three columns.
    """
    if np.PyArray_TYPE(a) != np.NPY_LONG or not np.PyArray_IS_C_CONTIGUOUS(a):
        raise ValueError('Expected a contiguous array of integers')
    if a.ndim == 2 and a.shape[1] != 3:
        raise ValueError('Expected an array of indices with three columns')
    if np.PyArray_SIZE(a) < n:
        raise IndexError('Expected an array of at least {0} values, got {1}'.format(n, np.PyArray_SIZE(a)))
    if np.PyArray_SIZE(a) == 0:
        return _empty_indices
    return <long *> np.PyArray_DATA(a)


cdef inline int _check_indices(long * indices, Py_ssize_t num_species) except -1:
    """
    Check that the three species `indices` of a reaction (or network) refer to core species, as the
    bounds checks of the arrays indexed by them did: the first index must be valid, and the others valid or -1.
    """
    if indices[0] < 0 or indices[0] >= num_species:
        raise IndexError('Invalid species index {0}'.format(indices[0]))
    if indices[1] < -1 or indices[1] >= num_species:
        raise IndexError('Invalid species index {0}'.format(indices[1]))
    if indices[2] < -1 or indices[2] >= num_species:
        raise IndexError('Invalid species index {0}'.format(indices[2]))
    return 0


def pairwise_sum(np.ndarray[np.float64_t, ndim=1, mode='c'] a):
    """
    Return the sum of the values of the contiguous array `a`, rounded exactly as ``np.sum(a)`` rounds it.
    """
    if a.shape[0] == 0:
        return 0.
    return _pairwise_sum(&a[0], a.shape[0])


cdef class SimpleReactor(ReactionSystem):
    """
    A reaction system consisting of a homogeneous, isothermal, isobaric batch
    reactor. These assumptions allow for a number of optimizations that enable
    this solver to complete very rapidly, even for large kinetic models.
    """

    cdef public ScalarQuantity T
    cdef public ScalarQuantity P
    cdef public double V
    cdef public bint constant_volume
    cdef public dict initial_mole_fractions
    cdef public list const_spc_names 
    cdef public list const_spc_indices

    # collider variables

    """
    pdep_collider_kinetics:
    an array that contains a reference to the kinetics object of the reaction
    that has pressure dependent kinetics.
    """
    cdef public list pdep_collider_kinetics

    """
    collider_efficiencies:
    an array consisting of array elements, each element corresponding to a reaction.
    Each element is an array with each position in the array corresponding to the collider efficiency
    of the core species. The collider efficiency is set to 1 if the species was not found in the list
    of colliders.
    """
    cdef public np.ndarray collider_efficiencies

    """
    pdep_collision_reaction_indices: 
    array that contains the indices of those reactions that 
    have pressure dependent kinetics. E.g. [4, 10, 2, 123]
    """
    cdef public np.ndarray pdep_collision_reaction_indices

    """
    pdep_specific_collider_kinetics:
    an array that contains a reference to the kinetics object of the reaction
    that has pressure dependent kinetics with a specific species as a third body collider.
    """
    cdef public list pdep_specific_collider_kinetics

    """
    specific_collider_species:
    a list that contains object references to species which are specific third body colliders
    in the respective reactions in pdep_specific_collider_reaction_indices.
    """
    cdef public list specific_collider_species

    """
    pdep_specific_collider_reaction_indices:
    an array that contains the indices of reactions that have
    a specifcCollider attribyte. E.g. [16, 155, 90]
    """
    cdef public np.ndarray pdep_specific_collider_reaction_indices

    # The parts of the rate coefficients of the reactions with collider efficiencies that only depend on the
    # temperature (see ThirdBody, Lindemann and Troe.get_temperature_terms()), which are calculated once per
    # simulation instead of in every residual evaluation, for the kinetics in collider_rate_terms_kinetics at the
    # temperature collider_rate_terms_T. collider_rate_kinds is 1, 2 or 3 for ThirdBody, Lindemann or Troe
    # kinetics, and 0 for other kinetics, whose rate coefficients are calculated as usual
    cdef np.ndarray collider_rate_terms
    cdef np.ndarray collider_rate_kinds
    cdef double collider_rate_terms_T
    cdef list collider_rate_terms_kinetics

    # The species index arrays (and numbers of core species and reactions) that residual() has checked to
    # contain valid core species indices. The arrays are created by initialize_model() and not modified
    # afterwards, so they only need to be checked once
    cdef object checked_reactant_indices
    cdef object checked_product_indices
    cdef object checked_network_indices
    cdef Py_ssize_t checked_num_core_species
    cdef Py_ssize_t checked_num_core_reactions

    cdef public dict sens_conditions

    cdef public list Trange
    cdef public list Prange
    cdef public int n_sims

    def __init__(self, T, P, initial_mole_fractions, n_sims=1, termination=None, sensitive_species=None,
                 sensitivity_threshold=1e-3, sens_conditions=None, const_spc_names=None):
        ReactionSystem.__init__(self, termination, sensitive_species, sensitivity_threshold)

        if type(T) != list:
            self.T = Quantity(T)
        else:
            self.Trange = [Quantity(t) for t in T]

        if type(P) != list:
            self.P = Quantity(P)
        else:
            self.Prange = [Quantity(p) for p in P]

        self.initial_mole_fractions = initial_mole_fractions

        #Constant Species Properties
        self.const_spc_indices = None
        self.const_spc_names = const_spc_names #store index of constant species 

        self.V = 0  # will be set in initialize_model
        self.constant_volume = False

        self.pdep_collision_reaction_indices = None
        self.pdep_collider_kinetics = None
        self.collider_efficiencies = None
        self.pdep_specific_collider_reaction_indices = None
        self.pdep_specific_collider_kinetics = None
        self.specific_collider_species = None
        self.sens_conditions = sens_conditions
        self.n_sims = n_sims

    def __reduce__(self):
        """
        A helper function used when pickling an object.
        """
        return (self.__class__,
                (self.T, self.P, self.initial_mole_fractions, self.n_sims, self.termination))

    def convert_initial_keys_to_species_objects(self, species_dict):
        """
        Convert the initial_mole_fractions dictionary from species names into species objects,
        using the given dictionary of species.
        """
        initial_mole_fractions = {}
        for label, moleFrac in self.initial_mole_fractions.items():
            initial_mole_fractions[species_dict[label]] = moleFrac
        self.initial_mole_fractions = initial_mole_fractions

        conditions = {}
        if self.sens_conditions is not None:
            for label, value in self.sens_conditions.items():
                if label == 'T' or label == 'P':
                    conditions[label] = value
                else:
                    conditions[species_dict[label]] = value
        self.sens_conditions = conditions

    def get_const_spc_indices(self, core_species):
        """
        Allow to identify constant Species position in solver
        """
        if self.const_spc_names is None:
            return
        if self.const_spc_indices is None:
            self.const_spc_indices = []
        else:
            return
        for name in self.const_spc_names:
            for spc in core_species:
                if spc.label == name:
                    self.const_spc_indices.append(core_species.index(spc))
                    break

    cpdef initialize_model(self, list core_species, list core_reactions, list edge_species, list edge_reactions,
                          list surface_species=None, list surface_reactions=None, list pdep_networks=None,
                          atol=1e-16, rtol=1e-8, sensitivity=False, sens_atol=1e-6, sens_rtol=1e-4,
                          filter_reactions=False, dict conditions=None):
        """
        Initialize a simulation of the simple reactor using the provided kinetic
        model.
        """

        if surface_species is None:
            surface_species = []
        if surface_reactions is None:
            surface_reactions = []

        # First call the base class version of the method
        # This initializes the attributes declared in the base class
        ReactionSystem.initialize_model(self, core_species=core_species, core_reactions=core_reactions,
                                       edge_species=edge_species, edge_reactions=edge_reactions,
                                       surface_species=surface_species, surface_reactions=surface_reactions,
                                       pdep_networks=pdep_networks, atol=atol, rtol=rtol, sensitivity=sensitivity,
                                       sens_atol=sens_atol, sens_rtol=sens_rtol, filter_reactions=filter_reactions,
                                       conditions=conditions)

        # Set initial conditions
        self.set_initial_conditions()

        # Compute reaction thresholds if reaction filtering is turned on
        if filter_reactions:
            ReactionSystem.set_initial_reaction_thresholds(self)

        self.set_colliders(core_reactions, edge_reactions, core_species)

        ReactionSystem.compute_network_variables(self, pdep_networks)

        # Generate forward and reverse rate coefficients k(T,P)
        self.generate_rate_coefficients(core_reactions, edge_reactions)

        ReactionSystem.set_initial_derivative(self)
        # Initialize the model
        ReactionSystem.initialize_solver(self)

    def calculate_effective_pressure(self, rxn):
        """
        Computes the effective pressure for a reaction as:

        .. math:: P_{eff} = P * \\sum_i \\frac{y_i * eff_i}{\\sum_j y_j}

        with:
            - P the pressure of the reactor,
            - y the array of initial moles of the core species

        or as:

        .. math:: P_{eff} = \\frac{P * y_{specific_collider}}{\\sum_j y_j}

        if a specific_collider is mentioned.
        """

        y0_core_species = self.y0[:self.num_core_species]
        sum_core_species = np.sum(y0_core_species)

        j = self.reaction_index[rxn]
        for i in range(self.pdep_collision_reaction_indices.shape[0]):
            if j == self.pdep_collision_reaction_indices[i]:
                # Calculate effective pressure
                if rxn.specific_collider is None:
                    Peff = self.P.value_si * np.sum(self.collider_efficiencies[i] * y0_core_species / sum_core_species)
                else:
                    logging.debug("Calculating Peff using {0} as a specific_collider".format(rxn.specific_collider))
                    Peff = self.P.value_si * self.y0[self.species_index[rxn.specific_collider]] / sum_core_species
                return Peff
        return self.P.value_si

    def generate_rate_coefficients(self, core_reactions, edge_reactions):
        """
        Populates the forward rate coefficients (kf), reverse rate coefficients (kb)
        and equilibrium constants (Keq) arrays with the values computed at the temperature
        and (effective) pressure of the reaction system.
        """

        # Compute the effective pressures as calculate_effective_pressure() does, without searching
        # the pressure-dependent reactions with collider efficiencies for each reaction
        y0_core_species = self.y0[:self.num_core_species]
        sum_core_species = np.sum(y0_core_species)
        collision_indices = {}
        for i in range(self.pdep_collision_reaction_indices.shape[0]):
            collision_indices.setdefault(int(self.pdep_collision_reaction_indices[i]), i)

        for rxn in itertools.chain(core_reactions, edge_reactions):
            j = self.reaction_index[rxn]
            i = collision_indices.get(j, -1)
            if i < 0:
                Peff = self.P.value_si
            elif rxn.specific_collider is None:
                Peff = self.P.value_si * np.sum(self.collider_efficiencies[i] * y0_core_species / sum_core_species)
            else:
                logging.debug("Calculating Peff using {0} as a specific_collider".format(rxn.specific_collider))
                Peff = self.P.value_si * self.y0[self.species_index[rxn.specific_collider]] / sum_core_species
            self.kf[j] = rxn.get_rate_coefficient(self.T.value_si, Peff)

            if rxn.reversible:
                self.Keq[j] = rxn.get_equilibrium_constant(self.T.value_si)
                self.kb[j] = self.kf[j] / self.Keq[j]
            else:
                self.kb[j] = 0.0
                self.Keq[j] = np.inf

    def get_threshold_rate_constants(self, model_settings):
        """
        Get the threshold rate constants for reaction filtering.
        """
        # Set the maximum unimolecular rate to be kB*T/h
        unimolecular_threshold_rate_constant = 2.08366122e10 * self.T.value_si
        # Set the maximum bi/trimolecular rate by using the user-defined rate constant threshold
        bimolecular_threshold_rate_constant = model_settings.filter_threshold
        # Maximum trimolecular rate constants are approximately three
        # orders of magnitude smaller (accounting for the unit
        # conversion from m^3/mol/s to m^6/mol^2/s) based on
        # extending the Smoluchowski equation to three molecules
        trimolecular_threshold_rate_constant = model_settings.filter_threshold / 1e3
        return (unimolecular_threshold_rate_constant,
                bimolecular_threshold_rate_constant,
                trimolecular_threshold_rate_constant)

    def set_colliders(self, core_reactions, edge_reactions, core_species):
        """
        Store collider efficiencies and reaction indices for pdep reactions that have collider efficiencies,
        and store specific collider indices
        """
        pdep_collider_reaction_indices = []
        self.pdep_collider_kinetics = []
        collider_efficiencies = []
        pdep_specific_collider_reaction_indices = []
        self.pdep_specific_collider_kinetics = []
        self.specific_collider_species = []

        for rxn in itertools.chain(core_reactions, edge_reactions):
            if rxn.kinetics.is_pressure_dependent():
                if rxn.kinetics.efficiencies:
                    j = self.reaction_index[rxn]
                    pdep_collider_reaction_indices.append(j)
                    self.pdep_collider_kinetics.append(rxn.kinetics)
                    collider_efficiencies.append(rxn.kinetics.get_effective_collider_efficiencies(core_species))
                if rxn.specific_collider:
                    pdep_specific_collider_reaction_indices.append(self.reaction_index[rxn])
                    self.pdep_specific_collider_kinetics.append(rxn.kinetics)
                    self.specific_collider_species.append(rxn.specific_collider)

        self.pdep_collision_reaction_indices = np.array(pdep_collider_reaction_indices, int)
        self.collider_efficiencies = np.array(collider_efficiencies, float)
        self.pdep_specific_collider_reaction_indices = np.array(pdep_specific_collider_reaction_indices, int)
        # The temperature-dependent terms of the rate coefficients are calculated again for the new kinetics
        self.collider_rate_terms_kinetics = None

    cdef int set_collider_rate_terms(self, double T) except -1:
        """
        Calculate the parts of the rate coefficients of the reactions with collider efficiencies that only depend
        on the temperature `T`.
        """
        cdef Py_ssize_t i, n
        cdef object kinetics
        cdef np.ndarray terms, kinds
        cdef double * termsp
        cdef long * kindsp
        n = len(self.pdep_collider_kinetics)
        terms = _zeros(4 * n)
        kinds = np.zeros(n, int)
        termsp = _float_data(terms, 4 * n)
        kindsp = _int_data(kinds, n)
        for i in range(n):
            kinetics = self.pdep_collider_kinetics[i]
            # Only for these exact classes, whose get_rate_coefficient() uses get_temperature_terms() and
            # get_rate_from_terms(), not for subclasses that might calculate their rate coefficients differently
            if type(kinetics) is ThirdBody:
                kindsp[i] = 1
                (<ThirdBody> kinetics).get_temperature_terms(T, termsp + 4 * i)
            elif type(kinetics) is Lindemann:
                kindsp[i] = 2
                (<Lindemann> kinetics).get_temperature_terms(T, termsp + 4 * i)
            elif type(kinetics) is Troe:
                kindsp[i] = 3
                (<Troe> kinetics).get_temperature_terms(T, termsp + 4 * i)
        self.collider_rate_terms = terms
        self.collider_rate_kinds = kinds
        self.collider_rate_terms_T = T
        self.collider_rate_terms_kinetics = self.pdep_collider_kinetics
        return 0

    def set_initial_conditions(self):
        """
        Sets the initial conditions of the rate equations that represent the 
        current reactor model.

        The volume is set to the value derived from the ideal gas law, using the 
        user-defined pressure, temperature, and the number of moles of initial species.

        The species moles array (y0) is set to the values stored in the
        initial mole fractions dictionary.

        The initial species concentration is computed and stored in the
        core_species_concentrations array.

        """

        ReactionSystem.set_initial_conditions(self)

        for spec, moleFrac in self.initial_mole_fractions.items():
            i = self.get_species_index(spec)
            self.y0[i] = moleFrac

        # Use ideal gas law to compute volume
        self.V = constants.R * self.T.value_si * np.sum(self.y0[:self.num_core_species]) / self.P.value_si  # volume in m^3
        for j in range(self.num_core_species):
            self.core_species_concentrations[j] = self.y0[j] / self.V

    @cython.boundscheck(False)
    def residual(self, double t, np.ndarray y, np.ndarray dydt,
                 np.ndarray[np.float64_t, ndim=1] senpar = np.zeros(1, float)):

        """
        Return the residual function for the governing DAE system for the
        simple reaction system.
        """
        cdef np.ndarray delta, res, equilibrium_constants, collider_efficiencies, collider_terms
        cdef np.ndarray core_species_concentrations, core_species_rates, core_reaction_rates, network_leak_rates
        cdef np.ndarray core_species_consumption_rates, core_species_production_rates, C
        cdef np.ndarray[np.float64_t, ndim=2] jacobian, dgdk
        cdef np.ndarray pdep_collider_reaction_indices, pdep_specific_collider_reaction_indices
        cdef list pdep_collider_kinetics, pdep_specific_collider_kinetics
        cdef Py_ssize_t num_core_species, num_core_reactions, num_edge_species, num_edge_reactions, num_pdep_networks
        cdef Py_ssize_t i, j, z, first, second, third, num_inet
        cdef double k, V, reaction_rate, f_reaction_rate, rev_reaction_rate, T, P, Peff, y_sum
        cdef KineticsModel kinetics_model
        cdef Py_ssize_t num_rate_coefficients
        cdef long * rate_kinds
        cdef double * rate_terms
        # Pointers to the data of the arrays used in the loops below. The arrays are checked to be contiguous
        # and of the right type by _float_data() and _int_data(); accessing them through pointers avoids
        # acquiring a buffer for each array in every residual evaluation
        cdef long * ir
        cdef long * ip
        cdef long * inet
        cdef long * collider_indices
        cdef double * yp
        cdef double * dydtp
        cdef double * kf
        cdef double * kr
        cdef double * knet
        cdef double * keq
        cdef double * efficiencies
        cdef double * terms
        cdef double * Cp
        cdef double * concentrations
        cdef double * species_rates
        cdef double * reaction_rates
        cdef double * consumption_rates
        cdef double * production_rates
        cdef double * leak_rates
        cdef double * deltap

        num_core_species = self.core_species_rates.shape[0]
        num_core_reactions = self.core_reaction_rates.shape[0]
        num_edge_species = self.edge_species_rates.shape[0]
        num_edge_reactions = self.edge_reaction_rates.shape[0]
        num_pdep_networks = self.network_leak_rates.shape[0]

        yp = _float_data(y, num_core_species)
        ir = _int_data(self.reactant_indices, 3 * num_core_reactions)
        ip = _int_data(self.product_indices, 3 * num_core_reactions)
        # The rate coefficients of the core reactions, followed by those of the edge reactions
        num_rate_coefficients = self.kf.shape[0]
        if self.kb.shape[0] != num_rate_coefficients:
            raise ValueError('Inconsistent numbers of forward and reverse rate coefficients')
        kf = _float_data(self.kf, max(num_rate_coefficients, num_core_reactions))
        kr = _float_data(self.kb, max(num_rate_coefficients, num_core_reactions))

        # The sum of the core species amounts, rounded as np.sum() rounds it
        y_sum = _pairwise_sum(yp, num_core_species)

        # Recalculate any forward and reverse rate coefficients that involve pdep collision efficiencies
        if self.pdep_collision_reaction_indices.shape[0] != 0:
            T = self.T.value_si
            P = self.P.value_si
            equilibrium_constants = self.Keq
            keq = _float_data(equilibrium_constants, num_rate_coefficients)
            pdep_collider_reaction_indices = self.pdep_collision_reaction_indices
            pdep_collider_kinetics = self.pdep_collider_kinetics
            collider_efficiencies = self.collider_efficiencies
            collider_indices = _int_data(pdep_collider_reaction_indices, pdep_collider_reaction_indices.shape[0])
            efficiencies = _float_data(collider_efficiencies,
                                       pdep_collider_reaction_indices.shape[0] * num_core_species)
            if collider_efficiencies.ndim != 2 or collider_efficiencies.shape[1] != num_core_species:
                raise ValueError('Collider efficiencies do not match the core species')
            # The effective pressure of each reaction is
            # P * np.sum(collider_efficiencies[i] * y_core_species / np.sum(y_core_species)),
            # which is calculated here without numpy calls, but with exactly the same rounding
            collider_terms = _zeros(num_core_species)
            terms = <double *> np.PyArray_DATA(collider_terms)
            if self.collider_rate_terms_kinetics is not pdep_collider_kinetics or self.collider_rate_terms_T != T:
                self.set_collider_rate_terms(T)
            rate_kinds = _int_data(self.collider_rate_kinds, pdep_collider_reaction_indices.shape[0])
            rate_terms = _float_data(self.collider_rate_terms, 4 * pdep_collider_reaction_indices.shape[0])
            for i in range(pdep_collider_reaction_indices.shape[0]):
                for z in range(num_core_species):
                    terms[z] = efficiencies[i * num_core_species + z] * yp[z] / y_sum
                Peff = P * _pairwise_sum(terms, num_core_species)
                j = collider_indices[i]
                if j < 0 or j >= num_rate_coefficients:
                    raise IndexError('Invalid pressure-dependent reaction index {0}'.format(j))
                kinetics_model = pdep_collider_kinetics[i]
                if rate_kinds[i] == 3:
                    kf[j] = (<Troe> kinetics_model).get_rate_from_terms(T, Peff, rate_terms + 4 * i)
                elif rate_kinds[i] == 2:
                    kf[j] = (<Lindemann> kinetics_model).get_rate_from_terms(T, Peff, rate_terms + 4 * i)
                elif rate_kinds[i] == 1:
                    kf[j] = (<ThirdBody> kinetics_model).get_rate_from_terms(T, Peff, rate_terms + 4 * i)
                else:
                    kf[j] = kinetics_model.get_rate_coefficient(T, Peff)
                kr[j] = kf[j] / keq[j]
        if self.pdep_specific_collider_reaction_indices.shape[0] != 0:
            T = self.T.value_si
            P = self.P.value_si
            equilibrium_constants = self.Keq
            keq = _float_data(equilibrium_constants, num_rate_coefficients)
            pdep_specific_collider_reaction_indices = self.pdep_specific_collider_reaction_indices
            pdep_specific_collider_kinetics = self.pdep_specific_collider_kinetics
            specific_collider_species = self.specific_collider_species
            collider_indices = _int_data(pdep_specific_collider_reaction_indices,
                                         pdep_specific_collider_reaction_indices.shape[0])
            for i in range(pdep_specific_collider_reaction_indices.shape[0]):
                j = collider_indices[i]
                if j < 0 or j >= num_rate_coefficients:
                    raise IndexError('Invalid pressure-dependent reaction index {0}'.format(j))
                z = self.species_index[specific_collider_species[i]]
                if len(y) > z:
                    # Calculate the effective pressure
                    Peff = P * yp[z] / y_sum
                    kinetics_model = pdep_specific_collider_kinetics[i]
                    kf[j] = kinetics_model.get_rate_coefficient(T, Peff)
                else:
                    kf[j] = 0
                kr[j] = kf[j] / keq[j]

        num_inet = self.network_indices.shape[0]
        inet = _int_data(self.network_indices, 3 * num_inet)
        knet = _float_data(self.network_leak_coefficients, num_inet)

        core_species_concentrations = _zeros(self.core_species_concentrations.shape[0])
        core_species_rates = _zeros(num_core_species)
        core_reaction_rates = _zeros(num_core_reactions)
        core_species_consumption_rates = _zeros(self.core_species_consumption_rates.shape[0])
        core_species_production_rates = _zeros(self.core_species_production_rates.shape[0])
        network_leak_rates = _zeros(num_pdep_networks)
        C = _zeros(self.core_species_concentrations.shape[0])
        if (core_species_concentrations.shape[0] < num_core_species
                or core_species_consumption_rates.shape[0] < num_core_species
                or core_species_production_rates.shape[0] < num_core_species
                or num_pdep_networks < num_inet):
            raise ValueError('Inconsistent array sizes in the reaction system')
        concentrations = <double *> np.PyArray_DATA(core_species_concentrations)
        species_rates = <double *> np.PyArray_DATA(core_species_rates)
        reaction_rates = <double *> np.PyArray_DATA(core_reaction_rates)
        consumption_rates = <double *> np.PyArray_DATA(core_species_consumption_rates)
        production_rates = <double *> np.PyArray_DATA(core_species_production_rates)
        leak_rates = <double *> np.PyArray_DATA(network_leak_rates)
        Cp = <double *> np.PyArray_DATA(C)

        if (self.checked_reactant_indices is not self.reactant_indices
                or self.checked_product_indices is not self.product_indices
                or self.checked_network_indices is not self.network_indices
                or self.checked_num_core_species != num_core_species
                or self.checked_num_core_reactions != num_core_reactions):
            # Check that the core reactions and networks only refer to core species (the arrays indexed by
            # these indices below are accessed through pointers, without bounds checks)
            for j in range(num_core_reactions):
                _check_indices(ir + 3 * j, num_core_species)
                _check_indices(ip + 3 * j, num_core_species)
            for j in range(num_inet):
                if inet[3 * j] != -1:
                    _check_indices(inet + 3 * j, num_core_species)
            self.checked_reactant_indices = self.reactant_indices
            self.checked_product_indices = self.product_indices
            self.checked_network_indices = self.network_indices
            self.checked_num_core_species = num_core_species
            self.checked_num_core_reactions = num_core_reactions

        # Use ideal gas law to compute volume
        V = constants.R * self.T.value_si * y_sum / self.P.value_si
        self.V = V

        for j in range(num_core_species):
            Cp[j] = yp[j] / V
            concentrations[j] = Cp[j]

        # Only the core reactions are needed to evaluate the residual. The edge reaction and species
        # rates are only needed after each step, so they are calculated from the concentrations of
        # the last residual evaluation by update_edge_rates()
        for j in range(num_core_reactions):
            k = kf[j]
            if ir[3 * j] >= num_core_species or ir[3 * j + 1] >= num_core_species or ir[3 * j + 2] >= num_core_species:
                f_reaction_rate = 0.0
            elif ir[3 * j + 1] == -1:  # only one reactant
                f_reaction_rate = k * Cp[ir[3 * j]]
            elif ir[3 * j + 2] == -1:  # only two reactants
                f_reaction_rate = k * Cp[ir[3 * j]] * Cp[ir[3 * j + 1]]
            else:  # three reactants
                f_reaction_rate = k * Cp[ir[3 * j]] * Cp[ir[3 * j + 1]] * Cp[ir[3 * j + 2]]
            k = kr[j]
            if ip[3 * j] >= num_core_species or ip[3 * j + 1] >= num_core_species or ip[3 * j + 2] >= num_core_species:
                rev_reaction_rate = 0.0
            elif ip[3 * j + 1] == -1:  # only one reactant
                rev_reaction_rate = k * Cp[ip[3 * j]]
            elif ip[3 * j + 2] == -1:  # only two reactants
                rev_reaction_rate = k * Cp[ip[3 * j]] * Cp[ip[3 * j + 1]]
            else:  # three reactants
                rev_reaction_rate = k * Cp[ip[3 * j]] * Cp[ip[3 * j + 1]] * Cp[ip[3 * j + 2]]

            reaction_rate = f_reaction_rate - rev_reaction_rate

            # Set the reaction and species rates
            # The reaction is a core reaction
            reaction_rates[j] = reaction_rate

            # Add/substract the total reaction rate from each species rate
            # Since it's a core reaction we know that all of its reactants
            # and products are core species
            first = ir[3 * j]
            species_rates[first] -= reaction_rate
            consumption_rates[first] += f_reaction_rate
            production_rates[first] += rev_reaction_rate
            second = ir[3 * j + 1]
            if second != -1:
                species_rates[second] -= reaction_rate
                consumption_rates[second] += f_reaction_rate
                production_rates[second] += rev_reaction_rate
                third = ir[3 * j + 2]
                if third != -1:
                    species_rates[third] -= reaction_rate
                    consumption_rates[third] += f_reaction_rate
                    production_rates[third] += rev_reaction_rate
            first = ip[3 * j]
            species_rates[first] += reaction_rate
            production_rates[first] += f_reaction_rate
            consumption_rates[first] += rev_reaction_rate
            second = ip[3 * j + 1]
            if second != -1:
                species_rates[second] += reaction_rate
                production_rates[second] += f_reaction_rate
                consumption_rates[second] += rev_reaction_rate
                third = ip[3 * j + 2]
                if third != -1:
                    species_rates[third] += reaction_rate
                    production_rates[third] += f_reaction_rate
                    consumption_rates[third] += rev_reaction_rate

        for j in range(num_inet):
            if inet[3 * j] != -1: #all source species are in the core
                k = knet[j]
                if inet[3 * j + 1] == -1:  # only one reactant
                    reaction_rate = k * Cp[inet[3 * j]]
                elif inet[3 * j + 2] == -1:  # only two reactants
                    reaction_rate = k * Cp[inet[3 * j]] * Cp[inet[3 * j + 1]]
                else:  # three reactants
                    reaction_rate = k * Cp[inet[3 * j]] * Cp[inet[3 * j + 1]] * Cp[inet[3 * j + 2]]
                leak_rates[j] = reaction_rate
            else:
                leak_rates[j] = 0.0

        if self.const_spc_indices is not None:
            for spc_index in self.const_spc_indices:
                core_species_rates[spc_index] = 0

        self.core_species_concentrations = core_species_concentrations
        self.core_species_rates = core_species_rates
        self.core_species_production_rates = core_species_production_rates
        self.core_species_consumption_rates = core_species_consumption_rates
        self.core_reaction_rates = core_reaction_rates
        self.edge_rate_concentrations = C
        self.network_leak_rates = network_leak_rates

        if self.sensitivity:
            res = core_species_rates * V
            delta = np.zeros(len(y), float)
            delta[:num_core_species] = res
            if self.jacobian_matrix is None:
                jacobian = self.jacobian(t, y, dydt, 0, senpar)
            else:
                jacobian = self.jacobian_matrix
            dgdk = ReactionSystem.compute_rate_derivative(self)
            for j in range(num_core_reactions + num_core_species):
                for i in range(num_core_species):
                    for z in range(num_core_species):
                        delta[(j + 1) * num_core_species + i] += jacobian[i, z] * y[(j + 1) * num_core_species + z]
                    delta[(j + 1) * num_core_species + i] += dgdk[i, j]
            delta = delta - dydt
        else:
            # delta = core_species_rates * V - dydt, without the temporary arrays
            dydtp = _float_data(dydt, num_core_species)
            if dydt.shape[0] != num_core_species:
                raise ValueError('operands could not be broadcast together with shapes ({0},) ({1},)'.format(
                    num_core_species, dydt.shape[0]))
            delta = _zeros(num_core_species)
            deltap = <double *> np.PyArray_DATA(delta)
            for i in range(num_core_species):
                deltap[i] = species_rates[i] * V
                deltap[i] = deltap[i] - dydtp[i]

        # Return DELTA, IRES.  IRES is set to 1 in order to tell DASPK to evaluate the sensitivity residuals
        return delta, 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, double t, np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] dydt,
                 double cj, np.ndarray[np.float64_t, ndim=1] senpar = np.zeros(1, float)):
        """
        Return the analytical Jacobian for the reaction system.
        """
        cdef np.ndarray[np.int_t, ndim=2] ir, ip
        cdef np.ndarray[np.float64_t, ndim=1] kf, kr, C
        # The matrix is C-contiguous, which lets the compiler vectorize the loops over its rows
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] pd
        cdef int num_core_reactions, num_core_species, i, j
        cdef double k, V, Ctot, deriv, corr

        ir = self.reactant_indices
        ip = self.product_indices

        kf = self.kf
        kr = self.kb
        num_core_reactions = len(self.core_reaction_rates)
        num_core_species = len(self.core_species_concentrations)

        pd = -cj * np.identity(num_core_species, float)

        V = constants.R * self.T.value_si * np.sum(y[:num_core_species]) / self.P.value_si

        Ctot = self.P.value_si / (constants.R * self.T.value_si)

        C = np.zeros_like(self.core_species_concentrations)
        for j in range(num_core_species):
            C[j] = y[j] / V

        for j in range(num_core_reactions):

            k = kf[j]
            if ir[j, 1] == -1:  # only one reactant
                deriv = k
                pd[ir[j, 0], ir[j, 0]] -= deriv

                pd[ip[j, 0], ir[j, 0]] += deriv
                if ip[j, 1] != -1:
                    pd[ip[j, 1], ir[j, 0]] += deriv
                    if ip[j, 2] != -1:
                        pd[ip[j, 2], ir[j, 0]] += deriv


            elif ir[j, 2] == -1:  # only two reactants
                corr = - k * C[ir[j, 0]] * C[ir[j, 1]] / Ctot
                if ir[j, 0] == ir[j, 1]:  # reactants are the same
                    deriv = 2 * k * C[ir[j, 0]]
                    pd[ir[j, 0], ir[j, 0]] -= 2 * deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] -= 2 * corr

                    pd[ip[j, 0], ir[j, 0]] += deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] += corr
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 0]] += deriv
                        for i in range(num_core_species):
                            pd[ip[j, 1], i] += corr
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 0]] += deriv
                            for i in range(num_core_species):
                                pd[ip[j, 2], i] += corr

                else:
                    # Derivative with respect to reactant 1
                    deriv = k * C[ir[j, 1]]
                    pd[ir[j, 0], ir[j, 0]] -= deriv
                    pd[ir[j, 1], ir[j, 0]] -= deriv

                    pd[ip[j, 0], ir[j, 0]] += deriv
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 0]] += deriv
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 0]] += deriv

                    # Derivative with respect to reactant 2
                    deriv = k * C[ir[j, 0]]
                    pd[ir[j, 0], ir[j, 1]] -= deriv
                    pd[ir[j, 1], ir[j, 1]] -= deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] -= corr
                        pd[ir[j, 1], i] -= corr

                    pd[ip[j, 0], ir[j, 1]] += deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] += corr
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 1]] += deriv
                        for i in range(num_core_species):
                            pd[ip[j, 1], i] += corr
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 1]] += deriv
                            for i in range(num_core_species):
                                pd[ip[j, 2], i] += corr


            else:  # three reactants
                corr = - 2 * k * C[ir[j, 0]] * C[ir[j, 1]] * C[ir[j, 2]] / Ctot
                if (ir[j, 0] == ir[j, 1] & ir[j, 0] == ir[j, 2]):
                    deriv = 3 * k * C[ir[j, 0]] * C[ir[j, 0]]
                    pd[ir[j, 0], ir[j, 0]] -= 3 * deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] -= 3 * corr

                    pd[ip[j, 0], ir[j, 0]] += deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] += corr
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 0]] += deriv
                        for i in range(num_core_species):
                            pd[ip[j, 1], i] += corr
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 0]] += deriv
                            for i in range(num_core_species):
                                pd[ip[j, 2], i] += corr

                elif ir[j, 0] == ir[j, 1]:
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ir[j, 0]] * C[ir[j, 2]]
                    pd[ir[j, 0], ir[j, 0]] -= 2 * deriv
                    pd[ir[j, 2], ir[j, 0]] -= deriv

                    pd[ip[j, 0], ir[j, 0]] += deriv
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 0]] += deriv
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 0]] += deriv

                    # derivative with respect to reactant 3
                    deriv = k * C[ir[j, 0]] * C[ir[j, 0]]
                    pd[ir[j, 0], ir[j, 2]] -= 2 * deriv
                    pd[ir[j, 2], ir[j, 2]] -= deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] -= 2 * corr
                        pd[ir[j, 2], i] -= corr

                    pd[ip[j, 0], ir[j, 2]] += deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] += corr
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 2]] += deriv
                        for i in range(num_core_species):
                            pd[ip[j, 1], i] += corr
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 2]] += deriv
                            for i in range(num_core_species):
                                pd[ip[j, 2], i] += corr


                elif ir[j, 1] == ir[j, 2]:
                    # derivative with respect to reactant 1
                    deriv = k * C[ir[j, 1]] * C[ir[j, 1]]
                    pd[ir[j, 0], ir[j, 0]] -= deriv
                    pd[ir[j, 1], ir[j, 0]] -= 2 * deriv

                    pd[ip[j, 0], ir[j, 0]] += deriv
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 0]] += deriv
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 0]] += deriv
                            # derivative with respect to reactant 2
                    deriv = 2 * k * C[ir[j, 0]] * C[ir[j, 1]]
                    pd[ir[j, 0], ir[j, 1]] -= deriv
                    pd[ir[j, 1], ir[j, 1]] -= 2 * deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] -= corr
                        pd[ir[j, 1], i] -= 2 * corr

                    pd[ip[j, 0], ir[j, 1]] += deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] += corr
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 1]] += deriv
                        for i in range(num_core_species):
                            pd[ip[j, 1], i] += corr
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 1]] += deriv
                            for i in range(num_core_species):
                                pd[ip[j, 2], i] += corr

                elif ir[j, 0] == ir[j, 2]:
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ir[j, 0]] * C[ir[j, 1]]
                    pd[ir[j, 0], ir[j, 0]] -= 2 * deriv
                    pd[ir[j, 1], ir[j, 0]] -= deriv

                    pd[ip[j, 0], ir[j, 0]] += deriv
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 0]] += deriv
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 0]] += deriv
                    # derivative with respect to reactant 2
                    deriv = k * C[ir[j, 0]] * C[ir[j, 0]]
                    pd[ir[j, 0], ir[j, 1]] -= 2 * deriv
                    pd[ir[j, 1], ir[j, 1]] -= deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] -= 2 * corr
                        pd[ir[j, 1], i] -= corr

                    pd[ip[j, 0], ir[j, 1]] += deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] += corr
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 1]] += deriv
                        for i in range(num_core_species):
                            pd[ip[j, 1], i] += corr
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 1]] += deriv
                            for i in range(num_core_species):
                                pd[ip[j, 2], i] += corr

                else:
                    # derivative with respect to reactant 1
                    deriv = k * C[ir[j, 1]] * C[ir[j, 2]]
                    pd[ir[j, 0], ir[j, 0]] -= deriv
                    pd[ir[j, 1], ir[j, 0]] -= deriv
                    pd[ir[j, 2], ir[j, 0]] -= deriv

                    pd[ip[j, 0], ir[j, 0]] += deriv
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 0]] += deriv
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 0]] += deriv

                            # derivative with respect to reactant 2
                    deriv = k * C[ir[j, 0]] * C[ir[j, 2]]
                    pd[ir[j, 0], ir[j, 1]] -= deriv
                    pd[ir[j, 1], ir[j, 1]] -= deriv
                    pd[ir[j, 2], ir[j, 1]] -= deriv

                    pd[ip[j, 0], ir[j, 1]] += deriv
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 1]] += deriv
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 1]] += deriv

                            # derivative with respect to reactant 3
                    deriv = k * C[ir[j, 0]] * C[ir[j, 1]]
                    pd[ir[j, 0], ir[j, 2]] -= deriv
                    pd[ir[j, 1], ir[j, 2]] -= deriv
                    pd[ir[j, 2], ir[j, 2]] -= deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] -= corr
                        pd[ir[j, 1], i] -= corr
                        pd[ir[j, 2], i] -= corr

                    pd[ip[j, 0], ir[j, 2]] += deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] += corr
                    if ip[j, 1] != -1:
                        pd[ip[j, 1], ir[j, 2]] += deriv
                        for i in range(num_core_species):
                            pd[ip[j, 1], i] += corr
                        if ip[j, 2] != -1:
                            pd[ip[j, 2], ir[j, 2]] += deriv
                            for i in range(num_core_species):
                                pd[ip[j, 2], i] += corr

            k = kr[j]
            if ip[j, 1] == -1:  # only one reactant
                deriv = k
                pd[ip[j, 0], ip[j, 0]] -= deriv

                pd[ir[j, 0], ip[j, 0]] += deriv
                if ir[j, 1] != -1:
                    pd[ir[j, 1], ip[j, 0]] += deriv
                    if ir[j, 2] != -1:
                        pd[ir[j, 2], ip[j, 0]] += deriv


            elif ip[j, 2] == -1:  # only two reactants
                corr = -k * C[ip[j, 0]] * C[ip[j, 1]] / Ctot
                if ip[j, 0] == ip[j, 1]:
                    deriv = 2 * k * C[ip[j, 0]]
                    pd[ip[j, 0], ip[j, 0]] -= 2 * deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] -= 2 * corr

                    pd[ir[j, 0], ip[j, 0]] += deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] += corr
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 0]] += deriv
                        for i in range(num_core_species):
                            pd[ir[j, 1], i] += corr
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 0]] += deriv
                            for i in range(num_core_species):
                                pd[ir[j, 2], i] += corr

                else:
                    # Derivative with respect to reactant 1
                    deriv = k * C[ip[j, 1]]
                    pd[ip[j, 0], ip[j, 0]] -= deriv
                    pd[ip[j, 1], ip[j, 0]] -= deriv

                    pd[ir[j, 0], ip[j, 0]] += deriv
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 0]] += deriv
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 0]] += deriv

                    # Derivative with respect to reactant 2
                    deriv = k * C[ip[j, 0]]
                    pd[ip[j, 0], ip[j, 1]] -= deriv
                    pd[ip[j, 1], ip[j, 1]] -= deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] -= corr
                        pd[ip[j, 1], i] -= corr

                    pd[ir[j, 0], ip[j, 1]] += deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] += corr
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 1]] += deriv
                        for i in range(num_core_species):
                            pd[ir[j, 1], i] += corr
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 1]] += deriv
                            for i in range(num_core_species):
                                pd[ir[j, 2], i] += corr


            else:  # three reactants
                corr = - 2 * k * C[ip[j, 0]] * C[ip[j, 1]] * C[ip[j, 2]] / Ctot
                if (ip[j, 0] == ip[j, 1] & ip[j, 0] == ip[j, 2]):
                    deriv = 3 * k * C[ip[j, 0]] * C[ip[j, 0]]
                    pd[ip[j, 0], ip[j, 0]] -= 3 * deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] -= 3 * corr

                    pd[ir[j, 0], ip[j, 0]] += deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] += corr
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 0]] += deriv
                        for i in range(num_core_species):
                            pd[ir[j, 1], i] += corr
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 0]] += deriv
                            for i in range(num_core_species):
                                pd[ir[j, 2], i] += corr

                elif ip[j, 0] == ip[j, 1]:
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ip[j, 0]] * C[ip[j, 2]]
                    pd[ip[j, 0], ip[j, 0]] -= 2 * deriv
                    pd[ip[j, 2], ip[j, 0]] -= deriv

                    pd[ir[j, 0], ip[j, 0]] += deriv
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 0]] += deriv
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 0]] += deriv
                    # derivative with respect to reactant 3
                    deriv = k * C[ip[j, 0]] * C[ip[j, 0]]
                    pd[ip[j, 0], ip[j, 2]] -= 2 * deriv
                    pd[ip[j, 2], ip[j, 2]] -= deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] -= 2 * corr
                        pd[ip[j, 2], i] -= corr

                    pd[ir[j, 0], ip[j, 2]] += deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] += corr
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 2]] += deriv
                        for i in range(num_core_species):
                            pd[ir[j, 1], i] += corr
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 2]] += deriv
                            for i in range(num_core_species):
                                pd[ir[j, 2], i] += corr


                elif ip[j, 1] == ip[j, 2]:
                    # derivative with respect to reactant 1
                    deriv = k * C[ip[j, 1]] * C[ip[j, 1]]
                    pd[ip[j, 0], ip[j, 0]] -= deriv
                    pd[ip[j, 1], ip[j, 0]] -= 2 * deriv

                    pd[ir[j, 0], ip[j, 0]] += deriv
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 0]] += deriv
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 0]] += deriv

                            # derivative with respect to reactant 2
                    deriv = 2 * k * C[ip[j, 0]] * C[ip[j, 1]]
                    pd[ip[j, 0], ip[j, 1]] -= deriv
                    pd[ip[j, 1], ip[j, 1]] -= 2 * deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] -= corr
                        pd[ip[j, 1], i] -= 2 * corr

                    pd[ir[j, 0], ip[j, 1]] += deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] += corr
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 1]] += deriv
                        for i in range(num_core_species):
                            pd[ir[j, 1], i] += corr
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 1]] += deriv
                            for i in range(num_core_species):
                                pd[ir[j, 2], i] += corr

                elif ip[j, 0] == ip[j, 2]:
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ip[j, 0]] * C[ip[j, 1]]
                    pd[ip[j, 0], ip[j, 0]] -= 2 * deriv
                    pd[ip[j, 1], ip[j, 0]] -= deriv

                    pd[ir[j, 0], ip[j, 0]] += deriv
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 0]] += deriv
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 0]] += deriv
                    # derivative with respect to reactant 2
                    deriv = k * C[ip[j, 0]] * C[ip[j, 0]]
                    pd[ip[j, 0], ip[j, 1]] -= 2 * deriv
                    pd[ip[j, 1], ip[j, 1]] -= deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] -= 2 * corr
                        pd[ip[j, 1], i] -= corr

                    pd[ir[j, 0], ip[j, 1]] += deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] += corr
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 1]] += deriv
                        for i in range(num_core_species):
                            pd[ir[j, 1], i] += corr
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 1]] += deriv
                            for i in range(num_core_species):
                                pd[ir[j, 2], i] += corr

                else:
                    # derivative with respect to reactant 1
                    deriv = k * C[ip[j, 1]] * C[ip[j, 2]]
                    pd[ip[j, 0], ip[j, 0]] -= deriv
                    pd[ip[j, 1], ip[j, 0]] -= deriv
                    pd[ip[j, 2], ip[j, 0]] -= deriv

                    pd[ir[j, 0], ip[j, 0]] += deriv
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 0]] += deriv
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 0]] += deriv

                            # derivative with respect to reactant 2
                    deriv = k * C[ip[j, 0]] * C[ip[j, 2]]
                    pd[ip[j, 0], ip[j, 1]] -= deriv
                    pd[ip[j, 1], ip[j, 1]] -= deriv
                    pd[ip[j, 2], ip[j, 1]] -= deriv

                    pd[ir[j, 0], ip[j, 1]] += deriv
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 1]] += deriv
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 1]] += deriv

                            # derivative with respect to reactant 3
                    deriv = k * C[ip[j, 0]] * C[ip[j, 1]]
                    pd[ip[j, 0], ip[j, 2]] -= deriv
                    pd[ip[j, 1], ip[j, 2]] -= deriv
                    pd[ip[j, 2], ip[j, 2]] -= deriv
                    for i in range(num_core_species):
                        pd[ip[j, 0], i] -= corr
                        pd[ip[j, 1], i] -= corr
                        pd[ip[j, 2], i] -= corr

                    pd[ir[j, 0], ip[j, 2]] += deriv
                    for i in range(num_core_species):
                        pd[ir[j, 0], i] += corr
                    if ir[j, 1] != -1:
                        pd[ir[j, 1], ip[j, 2]] += deriv
                        for i in range(num_core_species):
                            pd[ir[j, 1], i] += corr
                        if ir[j, 2] != -1:
                            pd[ir[j, 2], ip[j, 2]] += deriv
                            for i in range(num_core_species):
                                pd[ir[j, 2], i] += corr

        self.jacobian_matrix = pd + cj * np.identity(num_core_species, float)
        return pd
