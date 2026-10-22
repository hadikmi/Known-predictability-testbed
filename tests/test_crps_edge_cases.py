"""Regression checks for finite-size probabilistic ensemble verification."""

import numpy as np
import pytest

from knownpred.metrics import crps_ensemble


def test_single_member_empirical_crps_equals_absolute_error():
    """The standard empirical CRPS remains valid for one member."""
    observations = np.array([-1.0, 0.2, 2.0])
    members = np.array([[0.0], [0.2], [-2.0]])
    expected = np.abs(members[:, 0] - observations)
    np.testing.assert_allclose(crps_ensemble(observations, members), expected)


def test_fair_crps_rejects_single_member():
    """The finite-ensemble unbiased correction needs at least two members."""
    with pytest.raises(ValueError, match="at least two ensemble members"):
        crps_ensemble(np.array([1.0]), np.array([[0.0]]), fair=True)


def test_crps_rejects_empty_or_malformed_member_arrays():
    for members in (np.empty((1, 0)), np.array([0.0, 1.0])):
        with pytest.raises(ValueError, match="shape \\(N, M\\)"):
            crps_ensemble(np.array([0.0]), members)


def test_two_member_fair_crps_matches_pairwise_definition():
    """Check the exact fair-CRPS identity, including an ensemble with ties."""
    observations = np.array([1.0, -1.0, 0.5])
    members = np.array([[0.0, 2.0], [-2.0, 0.0], [1.0, 1.0]])
    n_members = members.shape[1]
    first_term = np.abs(members - observations[:, None]).mean(axis=1)
    all_pairs = np.abs(members[:, :, None] - members[:, None, :]).sum(axis=(1, 2))
    expected = first_term - 0.5 * all_pairs / (n_members * (n_members - 1))
    np.testing.assert_allclose(crps_ensemble(observations, members, fair=True), expected)
