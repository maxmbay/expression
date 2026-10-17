import numpy as np
import pytest

from cellresponse.data.parse import (
    normalize_compound_name,
    parse_drugname_drugconc,
    strip_marker,
    to_micromolar,
)


def test_parses_treatment_string_with_trailing_space():
    [t] = parse_drugname_drugconc("[('Erdafitinib ', 0.05, 'uM')]")
    assert t.name == "Erdafitinib "
    assert t.dose_uM == pytest.approx(0.05)


def test_vehicle_dose_is_zero():
    [t] = parse_drugname_drugconc("[('DMSO_TF', 0.0, 'uM')]")
    assert t.dose_uM == 0.0


def test_rejects_code_instead_of_evaluating_it():
    with pytest.raises(ValueError):
        parse_drugname_drugconc("__import__('os').system('true')")


@pytest.mark.parametrize("raw", ["[]", "[('A', 'x', 'uM')]", "[('A', 1.0)]", "[('A', nan, 'uM')]"])
def test_rejects_malformed_treatments(raw):
    with pytest.raises(ValueError):
        parse_drugname_drugconc(raw)


def test_unit_conversion():
    assert to_micromolar(500, "nM") == pytest.approx(0.5)
    assert to_micromolar(5, "uM") == 5
    with pytest.raises(ValueError):
        to_micromolar(1, "mg/ml")


def test_name_normalization_catches_case_and_whitespace_aliases():
    assert normalize_compound_name("Erdafitinib ") == normalize_compound_name("erdafitinib")
    assert normalize_compound_name("Venetoclax") == normalize_compound_name("venetoclax")


def test_strip_marker_removes_leading_entry():
    genes, counts = strip_marker(np.array([1, 5, 19]), np.array([-2.0, 1.0, 3.0]))
    assert genes.tolist() == [5, 19]
    assert counts.tolist() == [1.0, 3.0]


@pytest.mark.parametrize(
    "genes,counts",
    [
        ([5, 19], [1.0, 3.0]),  # marker missing: must not drop a real gene
        ([1, 5, 19], [-2.0, -1.0, 3.0]),  # negative count after marker
        ([1, 5, 19], [-2.0, np.inf, 3.0]),
        ([1, 5, 19], [-2.0, 1.5, 3.0]),
        ([1, 5, 5], [-2.0, 1.0, 3.0]),
        ([1, 5], [-2.0, 1.0, 3.0]),
    ],
)
def test_strip_marker_rejects_invalid_rows(genes, counts):
    with pytest.raises(ValueError):
        strip_marker(np.array(genes), np.array(counts))
