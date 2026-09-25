"""Pure signed-stability margin primitives for R3.1."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry


SIGN_CONVENTION = (
    "negative=stable_below_leave-one-candidate-out hull; "
    "zero=boundary or elemental special case; positive=unstable above hull"
)
OFFICIAL_METHOD = (
    "pymatgen.PhaseDiagram.get_decomp_and_phase_separation_energy"
    "(stable_only=False)"
)
EXPLICIT_METHOD = (
    "remove-candidate_then_PhaseDiagram.get_decomp_and_e_above_hull"
    "(allow_negative=True,check_stable=False)"
)


@dataclass(frozen=True)
class DecompositionComponent:
    entry_id: str
    formula: str
    amount: float


@dataclass(frozen=True)
class SignedMarginResult:
    signed_energy_eV_per_atom: float | None
    stability_margin_eV_per_atom: float | None
    is_stable_by_tolerance: bool | None
    official_signed_energy_eV_per_atom: float | None
    explicit_loo_energy_eV_per_atom: float | None
    official_decomposition: tuple[DecompositionComponent, ...]
    explicit_loo_decomposition: tuple[DecompositionComponent, ...]
    method: str
    pymatgen_version: str
    sign_convention: str
    solver_status: str
    validation_comparable: bool
    validation_exception: str | None
    absolute_validation_error_eV_per_atom: float | None
    reconstructed_energy_above_hull_eV_per_atom: float | None
    full_hull_reconciliation_error_eV_per_atom: float | None
    same_composition_entry_count: int
    official_error: str | None
    explicit_loo_error: str | None


@lru_cache(maxsize=1)
def _version() -> str:
    try:
        return importlib.metadata.version("pymatgen")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _composition_signature(entry: ComputedEntry) -> tuple[tuple[str, float], ...]:
    fractional = entry.composition.fractional_composition.get_el_amt_dict()
    return tuple(sorted((str(element), round(float(amount), 12)) for element, amount in fractional.items()))


def _components(decomposition: dict[ComputedEntry, float] | None) -> tuple[DecompositionComponent, ...]:
    if not decomposition:
        return ()
    return tuple(
        DecompositionComponent(
            entry_id=str(entry.entry_id),
            formula=entry.composition.reduced_formula,
            amount=float(amount),
        )
        for entry, amount in sorted(
            decomposition.items(), key=lambda item: (str(item[0].entry_id), float(item[1]))
        )
    )


def compute_signed_margin(
    entries: Iterable[ComputedEntry],
    candidate_entry_id: str,
    *,
    stored_energy_above_hull_eV_per_atom: float | None = None,
    numerical_tolerance_eV_per_atom: float = 1e-8,
) -> SignedMarginResult:
    """Compute official and explicit leave-one-candidate-out signed energies.

    The official pymatgen value is always retained. For duplicate-composition
    polymorphs, pymatgen's phase-separation API intentionally excludes all
    same-composition entries from its competitor set, whereas explicit LOO
    retains the other polymorphs. Those rows are marked as documented semantic
    exceptions and the explicit LOO value supplies the formal hull-stability
    sign. Elemental endpoints without an alternative terminal are explicit
    special cases with an official value of zero.
    """

    materialized = tuple(entries)
    by_id = {str(entry.entry_id): entry for entry in materialized}
    if len(by_id) != len(materialized):
        raise ValueError("entry_id values must be unique inside one exact context")
    if candidate_entry_id not in by_id:
        raise KeyError(f"candidate entry is absent from context: {candidate_entry_id}")
    candidate = by_id[candidate_entry_id]
    phase_diagram = PhaseDiagram(materialized)

    official_value: float | None = None
    official_decomposition: dict[ComputedEntry, float] | None = None
    official_error: str | None = None
    try:
        official_decomposition, raw_official = phase_diagram.get_decomp_and_phase_separation_energy(
            candidate,
            stable_only=False,
            tols=(1e-10, 1e-8),
            maxiter=2000,
        )
        official_value = None if raw_official is None else float(raw_official)
    except Exception as exc:  # pymatgen exposes multiple solver exception types
        official_error = f"{type(exc).__name__}: {exc}"

    reconstructed_eah: float | None = None
    full_hull_error: float | None = None
    try:
        _, raw_eah = phase_diagram.get_decomp_and_e_above_hull(
            candidate, check_stable=True, on_error="raise"
        )
        reconstructed_eah = float(raw_eah)
        if stored_energy_above_hull_eV_per_atom is not None:
            full_hull_error = abs(
                reconstructed_eah - float(stored_energy_above_hull_eV_per_atom)
            )
    except Exception:
        reconstructed_eah = None

    explicit_value: float | None = None
    explicit_decomposition: dict[ComputedEntry, float] | None = None
    explicit_error: str | None = None
    remaining = [entry for entry in materialized if entry is not candidate]
    try:
        if not remaining:
            raise ValueError("no remaining entries after candidate removal")
        leave_one_out = PhaseDiagram(remaining)
        explicit_decomposition, raw_explicit = leave_one_out.get_decomp_and_e_above_hull(
            candidate,
            allow_negative=True,
            check_stable=False,
            on_error="raise",
        )
        explicit_value = float(raw_explicit)
    except Exception as exc:
        explicit_error = f"{type(exc).__name__}: {exc}"

    candidate_signature = _composition_signature(candidate)
    same_composition_count = sum(
        _composition_signature(entry) == candidate_signature for entry in materialized
    )
    is_elemental = len(candidate.composition.elements) == 1

    validation_exception: str | None = None
    validation_comparable = official_value is not None and explicit_value is not None
    if same_composition_count > 1:
        validation_comparable = False
        validation_exception = "PYMATGEN_SAME_COMPOSITION_EXCLUSION_SEMANTICS"
    elif explicit_value is None and is_elemental:
        validation_comparable = False
        validation_exception = "ELEMENTAL_ENDPOINT_WITHOUT_ALTERNATIVE_TERMINAL"
    elif explicit_value is None:
        validation_comparable = False
        validation_exception = "EXPLICIT_LOO_UNAVAILABLE"
    elif official_value is None:
        validation_comparable = False
        validation_exception = "OFFICIAL_API_UNAVAILABLE"

    signed_value = explicit_value if explicit_value is not None else official_value
    absolute_error = (
        abs(float(official_value) - float(explicit_value))
        if official_value is not None and explicit_value is not None
        else None
    )
    if signed_value is None:
        status = "SOLVER_ERROR"
    elif validation_exception == "PYMATGEN_SAME_COMPOSITION_EXCLUSION_SEMANTICS":
        status = "PASS_POLYMORPH_SEMANTIC_EXCEPTION"
    elif validation_exception == "ELEMENTAL_ENDPOINT_WITHOUT_ALTERNATIVE_TERMINAL":
        status = "PASS_ELEMENTAL_ENDPOINT_SPECIAL"
    elif validation_comparable:
        status = "PASS_COMPARABLE"
    else:
        status = "PASS_WITH_DOCUMENTED_EXCEPTION"

    return SignedMarginResult(
        signed_energy_eV_per_atom=signed_value,
        stability_margin_eV_per_atom=None if signed_value is None else abs(signed_value),
        is_stable_by_tolerance=(
            None
            if signed_value is None
            else signed_value <= numerical_tolerance_eV_per_atom
        ),
        official_signed_energy_eV_per_atom=official_value,
        explicit_loo_energy_eV_per_atom=explicit_value,
        official_decomposition=_components(official_decomposition),
        explicit_loo_decomposition=_components(explicit_decomposition),
        method=f"formal={EXPLICIT_METHOD}; raw_official={OFFICIAL_METHOD}",
        pymatgen_version=_version(),
        sign_convention=SIGN_CONVENTION,
        solver_status=status,
        validation_comparable=validation_comparable,
        validation_exception=validation_exception,
        absolute_validation_error_eV_per_atom=absolute_error,
        reconstructed_energy_above_hull_eV_per_atom=reconstructed_eah,
        full_hull_reconciliation_error_eV_per_atom=full_hull_error,
        same_composition_entry_count=same_composition_count,
        official_error=official_error,
        explicit_loo_error=explicit_error,
    )
