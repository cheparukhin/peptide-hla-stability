"""Stage 8: FoldX empirical interaction energy over the stage 4c folds.

**FoldX is not installed and cannot be installed by this code.** It is
proprietary: the binary and its per-user licence have to be downloaded by the
account holder from <https://foldxsuite.crg.eu/>. Everything in this module is
the part that does not need it -- conversion, validation, command construction,
output parsing, and the feature contract -- so that the run starts the moment
the binary lands. Nothing here downloads, patches, or works around a licence.

Four things this module is deliberately careful about.

1. **It does not re-implement fold discovery.** ``discover_folds`` and
   ``is_excluded`` are imported from :mod:`pepstab.structural_features`, which
   owns the ``_smoke`` / ``_shards`` / ``_failed`` exclusion. A second copy of
   that rule is a second chance to ingest harness folds as cohort rows.
   :func:`pepstab.structural_features.load_fold` also supplies the *verified*
   chain mapping -- which chain is the peptide, which is the HLA -- so this
   module never guesses it from chain letters.

2. **mmCIF -> PDB is validated, not assumed.** FoldX reads only PDB and is
   unforgiving of malformed input. :func:`cif_to_pdb` writes the file and
   :func:`validate_pdb` reads it back and compares it, atom for atom, against
   the mmCIF it came from: chain ids, residue numbering, residue names, atom
   names, elements, coordinates within the format's own precision, and the
   peptide still a separate chain whose sequence matches the metadata.

3. **The .fxout parser is header-driven.** FoldX's column set differs between
   releases, so :func:`parse_fxout` reads the names out of the file rather
   than from a hard-coded list. The exact header has **not** been seen on a
   real run here, because there is no binary to run; :func:`parse_fxout` is
   tested against the documented FoldX 5 ``AnalyseComplex`` layout and must be
   re-checked against the first real output (see
   ``reports/stage8_foldx.md`` section "First real output").

4. **The quantity is named for what it is.** FoldX estimates an equilibrium
   interaction free energy, in kcal/mol. The label is a dissociation half-life,
   governed by the barrier to unbinding. Columns are prefixed ``foldx_`` and
   the interaction term is ``foldx_interaction_energy``; nothing here is named
   "stability".
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# Shared API, imported rather than copied -- see the module docstring.
from .structural_features import (  # noqa: F401
    EXCLUDED_DIR_NAMES,
    GROOVE_RESIDUES,
    MappingError,
    discover_folds,
    is_excluded,
    load_fold,
)

__all__ = [
    "ConversionError",
    "FoldxNotInstalled",
    "FOLDX_REQUIRED_FILES",
    "cif_to_pdb",
    "validate_pdb",
    "prepare_pdb",
    "locate_foldx",
    "repair_command",
    "analyse_complex_command",
    "parse_fxout",
    "interaction_features",
    "score_fold",
    "discover_folds",
    "is_excluded",
]


class ConversionError(ValueError):
    """The PDB written from an mmCIF does not faithfully reproduce it."""


class FoldxNotInstalled(RuntimeError):
    """No FoldX binary / licence was found where the runner expected one."""


#: Coordinates survive the mmCIF -> PDB round trip only to PDB's three decimal
#: places, and B-factors to two. These are format precision, not tolerance for
#: a wrong atom: anything above them means atoms moved or were reordered.
COORD_TOLERANCE_A = 0.001
BFACTOR_TOLERANCE = 0.01

#: What an unpacked FoldX distribution has to provide. The binary name varies
#: by build (``foldx``, ``foldx_20251231``, ``foldx5MacC11``...), so the runner
#: globs for it rather than pinning one name; the support directory likewise
#: differs between FoldX 4 (``rotabase.txt``) and FoldX 5 (``molecules/``).
FOLDX_REQUIRED_FILES = ("the foldx executable", "rotabase.txt or molecules/")

#: Prefix on every column this module produces, so a feature table can never
#: confuse a FoldX energy with a Boltz confidence or a geometric feature.
FEATURE_PREFIX = "foldx_"


# ---------------------------------------------------------------------------
# mmCIF -> PDB
# ---------------------------------------------------------------------------


def _read_cif_atoms(cif_path: Path):
    """Heavy atoms of model 1, in file order, with B-factors.

    Same reading convention as ``pepstab.structural_features.load_fold`` and
    ``scripts/boltz_pose_check.py``: model 1, hetero dropped, hydrogens
    dropped. Boltz-2 writes no hydrogens and no hetero atoms, so in practice
    these drop nothing -- they are here so the convention matches the module
    that verified the chain mapping.
    """
    import numpy as np
    from biotite.structure.io.pdbx import CIFFile, get_structure

    atoms = get_structure(CIFFile.read(str(cif_path)), model=1, extra_fields=["b_factor"])
    if "hetero" in atoms.get_annotation_categories():
        atoms = atoms[~atoms.hetero]
    return atoms[~np.isin(atoms.element, ("H", "D"))]


def _insert_ter_records(text: str) -> str:
    """Add a ``TER`` after each chain and a final ``END``.

    biotite writes neither. FoldX keys on the chain-id column, so a missing
    ``TER`` is not fatal, but a chain break that is only implied by a column
    is exactly the kind of thing an empirical force field gets wrong quietly:
    without ``TER`` a parser is free to read the last residue of chain A and
    the first of chain B as bonded, which would invent a peptide bond across
    the HLA/beta2m boundary.
    """
    out: list[str] = []
    serial = 0
    previous: tuple[str, str, str] | None = None  # chain, resseq, resname
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            chain = line[21]
            resseq = line[22:27]
            resname = line[17:20]
            if previous is not None and chain != previous[0]:
                serial += 1
                out.append(
                    f"TER   {serial:5d}      {previous[2]:>3} {previous[0]}{previous[1]}"
                )
            serial += 1
            out.append(line)
            previous = (chain, resseq, resname)
        elif line.startswith(("TER", "END")):
            continue
        else:
            out.append(line)
    if previous is not None:
        serial += 1
        out.append(f"TER   {serial:5d}      {previous[2]:>3} {previous[0]}{previous[1]}")
    out.append("END")
    return "\n".join(out) + "\n"


def cif_to_pdb(cif_path: Path | str, pdb_path: Path | str) -> dict[str, Any]:
    """Write the mmCIF's model 1 as a PDB file FoldX can read.

    Returns a small report: atom and residue counts, the chain layout, and
    whether anything had to be dropped. Call :func:`validate_pdb` on the
    result -- this function writes, it does not certify.
    """
    from biotite.structure.io.pdb import PDBFile

    cif_path, pdb_path = Path(cif_path), Path(pdb_path)
    atoms = _read_cif_atoms(cif_path)

    if atoms.array_length() == 0:
        raise ConversionError(f"{cif_path}: no heavy atoms in model 1")
    if atoms.array_length() > 99999:
        # PDB's atom serial field is five columns. Our construct is ~3,155
        # atoms, so this is a tripwire for a future construct change, not a
        # live concern.
        raise ConversionError(
            f"{cif_path}: {atoms.array_length()} atoms exceeds the PDB serial field"
        )
    bad_chains = sorted({c for c in map(str, atoms.chain_id) if len(c) != 1})
    if bad_chains:
        raise ConversionError(
            f"{cif_path}: chain ids {bad_chains} are not single characters; "
            "PDB has one column for the chain and FoldX selects groups by it"
        )
    if int(atoms.res_id.max()) > 9999 or int(atoms.res_id.min()) < -999:
        raise ConversionError(f"{cif_path}: residue numbers do not fit the PDB field")

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    handle = PDBFile()
    handle.set_structure(atoms)
    pdb_path.write_text(_insert_ter_records("\n".join(handle.lines)))

    runs = _chain_runs(atoms)
    return {
        "cif": str(cif_path),
        "pdb": str(pdb_path),
        "n_atoms": int(atoms.array_length()),
        "chain_runs": runs,
        "n_residues": sum(n for _, n in runs),
        "bytes": pdb_path.stat().st_size,
    }


def _chain_runs(atoms) -> list[tuple[str, int]]:
    """Consecutive runs of chain id, counted in residues, in file order."""
    import biotite.structure as struc

    starts = struc.get_residue_starts(atoms)
    runs: list[tuple[str, int]] = []
    for cid in (str(c) for c in atoms.chain_id[starts]):
        if runs and runs[-1][0] == cid:
            runs[-1] = (cid, runs[-1][1] + 1)
        else:
            runs.append((cid, 1))
    return runs


def validate_pdb(
    cif_path: Path | str,
    pdb_path: Path | str,
    peptide_chain: str | None = None,
    peptide_sequence: str | None = None,
) -> dict[str, Any]:
    """Read the PDB back and prove it reproduces the mmCIF.

    FoldX is sensitive to malformed PDB and silent about some of it, so this
    checks rather than assumes. Raises :class:`ConversionError` on any
    disagreement; returns the measured deviations on success.
    """
    from biotite.structure.io.pdb import PDBFile

    cif_path, pdb_path = Path(cif_path), Path(pdb_path)
    source = _read_cif_atoms(cif_path)
    text = pdb_path.read_text()
    back = PDBFile.read(str(pdb_path)).get_structure(model=1, extra_fields=["b_factor"])

    if back.array_length() != source.array_length():
        raise ConversionError(
            f"{pdb_path}: {back.array_length()} atoms read back, "
            f"{source.array_length()} in {cif_path.name}"
        )
    for field in ("chain_id", "res_id", "res_name", "atom_name", "element"):
        got = np.asarray(getattr(back, field)).astype(str)
        want = np.asarray(getattr(source, field)).astype(str)
        if not (got == want).all():
            where = int(np.flatnonzero(got != want)[0])
            raise ConversionError(
                f"{pdb_path}: {field} differs at atom {where}: "
                f"{got[where]!r} vs {want[where]!r} in the mmCIF"
            )
    coord_dev = float(np.abs(back.coord - source.coord).max())
    if coord_dev > COORD_TOLERANCE_A:
        raise ConversionError(
            f"{pdb_path}: coordinates moved by up to {coord_dev:.4f} A, above the "
            f"{COORD_TOLERANCE_A} A the PDB format's three decimals can explain"
        )
    b_dev = float(np.abs(back.b_factor - source.b_factor).max())
    if b_dev > BFACTOR_TOLERANCE:
        raise ConversionError(f"{pdb_path}: B-factor (pLDDT) deviation {b_dev:.4f}")

    runs = _chain_runs(back)
    chains = [cid for cid, _ in runs]
    if len(chains) != len(set(chains)):
        raise ConversionError(
            f"{pdb_path}: chain ids {chains} are not contiguous blocks; FoldX "
            "would read one chain as two groups"
        )

    report: dict[str, Any] = {
        "n_atoms": int(back.array_length()),
        "chain_runs": runs,
        "max_coord_dev_A": coord_dev,
        "max_bfactor_dev": b_dev,
        "ter_records": text.count("\nTER") + text.startswith("TER"),
        "ends_with_END": text.rstrip().endswith("END"),
    }

    if peptide_chain is not None:
        if peptide_chain not in chains:
            raise ConversionError(
                f"{pdb_path}: peptide chain {peptide_chain!r} absent from {chains}"
            )
        sub = back[back.chain_id == peptide_chain]
        got = _one_letter(sub)
        report["peptide_chain"] = peptide_chain
        report["peptide_sequence"] = got
        if peptide_sequence is not None and got != peptide_sequence:
            raise ConversionError(
                f"{pdb_path}: chain {peptide_chain} is {got!r}, not the declared "
                f"peptide {peptide_sequence!r}"
            )
    if report["ter_records"] != len(chains):
        raise ConversionError(
            f"{pdb_path}: {report['ter_records']} TER records for {len(chains)} chains"
        )
    return report


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def _one_letter(atoms) -> str:
    import biotite.structure as struc

    starts = struc.get_residue_starts(atoms)
    return "".join(_THREE_TO_ONE.get(str(r).upper(), "X") for r in atoms.res_name[starts])


def prepare_pdb(folder: Path | str, out_dir: Path | str) -> dict[str, Any]:
    """Convert one verified fold directory to a validated PDB.

    The chain mapping comes from :func:`pepstab.structural_features.load_fold`,
    which re-derives every boundary from the mmCIF and raises on disagreement.
    This module therefore never decides which chain is the peptide from its
    letter; it is told, by the code that proved it.
    """
    folder, out_dir = Path(folder), Path(out_dir)
    fold = load_fold(folder)
    cif = next(iter(sorted(folder.glob("*.cif"))))
    name = folder.name
    pdb = Path(out_dir) / f"{name}.pdb"
    conversion = cif_to_pdb(cif, pdb)
    validation = validate_pdb(
        cif,
        pdb,
        peptide_chain=fold.peptide.chain_id,
        peptide_sequence=fold.metadata["inputs"]["peptide"],
    )
    return {
        "complex_id": folder.name,
        "allele": fold.metadata["inputs"]["allele"],
        "peptide": fold.metadata["inputs"]["peptide"],
        "pdb": str(pdb),
        "pdb_stem": pdb.stem,
        "hla_chain": fold.hla.chain_id,
        "peptide_chain": fold.peptide.chain_id,
        "beta2m_chain": None if fold.beta2m is None else fold.beta2m.chain_id,
        "hla_residues": fold.hla.length,
        "conversion": conversion,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# FoldX invocation
# ---------------------------------------------------------------------------


def locate_foldx(root: Path | str) -> dict[str, str]:
    """Find the executable and its support data in an unpacked distribution.

    Raises :class:`FoldxNotInstalled` with what is missing, because "FoldX is
    not here" is the expected state until the user supplies it, and a runner
    that says so precisely is worth more than one that says ``FileNotFound``.
    """
    root = Path(root)
    if not root.exists():
        raise FoldxNotInstalled(
            f"{root} does not exist. Unpack the FoldX distribution there; it must "
            f"contain {' and '.join(FOLDX_REQUIRED_FILES)}."
        )
    candidates = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name.lower().startswith("foldx") and p.suffix.lower() not in
        {".txt", ".pdf", ".md", ".cfg", ".tar", ".zip", ".gz"}
    ]
    if not candidates:
        raise FoldxNotInstalled(
            f"no file named foldx* under {root}. The binary name varies by build "
            "(foldx, foldx_<date>, foldx5...); unpack the whole archive, do not "
            "cherry-pick files."
        )
    if len(candidates) > 1:
        raise FoldxNotInstalled(
            f"{len(candidates)} foldx* executables under {root}: "
            f"{[p.name for p in candidates]}. Which build produced a number is "
            "provenance, so pick one rather than letting sort order decide."
        )
    binary = candidates[0]
    rotabase = next(iter(sorted(root.rglob("rotabase.txt"))), None)
    molecules = next((p for p in sorted(root.rglob("molecules")) if p.is_dir()), None)
    if rotabase is None and molecules is None:
        raise FoldxNotInstalled(
            f"{root} has a binary but neither rotabase.txt (FoldX 4) nor a "
            "molecules/ directory (FoldX 5). FoldX reads its rotamer and "
            "fragment data from one of those and exits without it."
        )
    return {
        "binary": str(binary),
        "rotabase": str(rotabase) if rotabase else "",
        "molecules": str(molecules) if molecules else "",
        "root": str(root),
        "sha256": _sha256(binary),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_command(binary: str, pdb: str, pdb_dir: str, out_dir: str) -> list[str]:
    """``RepairPDB``: rebuild clashing side chains before scoring.

    Whether this is needed is an empirical question the pilot answers, not an
    assumption -- it is the step that dominates runtime, and the input here is
    a predicted model with complete side chains rather than a crystal with
    missing ones. See ``reports/stage8_foldx.md``.
    """
    return [
        binary,
        "--command=RepairPDB",
        f"--pdb={pdb}",
        f"--pdb-dir={pdb_dir}",
        f"--output-dir={out_dir}",
    ]


def analyse_complex_command(
    binary: str, pdb: str, pdb_dir: str, out_dir: str, groups: Sequence[str]
) -> list[str]:
    """``AnalyseComplex`` between two chain groups, e.g. ``("C", "A")``.

    ``--analyseComplexChains`` takes the two groups to separate. Chains not
    named still sit in the file and contribute their environment, which is why
    beta2m stays in the PDB rather than being stripped.
    """
    if len(groups) != 2:
        raise ValueError(f"AnalyseComplex compares exactly two groups, got {groups!r}")
    return [
        binary,
        "--command=AnalyseComplex",
        f"--pdb={pdb}",
        f"--pdb-dir={pdb_dir}",
        f"--output-dir={out_dir}",
        f"--analyseComplexChains={groups[0]},{groups[1]}",
    ]


# ---------------------------------------------------------------------------
# .fxout parsing
# ---------------------------------------------------------------------------

_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _snake(name: str) -> str:
    """``Van der Waals clashes`` -> ``van_der_waals_clashes``."""
    out = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
    return re.sub(r"_+", "_", out)


def parse_fxout(text: str) -> list[dict[str, Any]]:
    """Parse a FoldX ``.fxout`` table into rows keyed by snake_case column name.

    **Header-driven on purpose.** FoldX's column set differs between releases
    and between commands, so the names come out of the file. A positional
    parser against a remembered FoldX 5 header would mislabel every term if the
    user's build differs by one column -- and mislabelled energy terms are the
    failure mode that survives into a results table looking fine.

    The file is a free-text preamble, then one tab-separated header line whose
    first field is ``Pdb``, then one row per analysed pair.
    """
    lines = [ln.rstrip("\n\r") for ln in text.splitlines()]
    header_at = None
    for i, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) > 2 and fields[0].strip().lower() == "pdb":
            header_at = i
            break
    if header_at is None:
        raise ValueError(
            "no FoldX table header found: expected a tab-separated line starting "
            "with 'Pdb'. First 400 characters:\n" + text[:400]
        )
    columns = [_snake(c) for c in lines[header_at].split("\t")]
    rows: list[dict[str, Any]] = []
    for line in lines[header_at + 1 :]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != len(columns):
            raise ValueError(
                f"FoldX row has {len(fields)} fields against {len(columns)} header "
                f"columns; the table is not the shape the header declares:\n{line!r}"
            )
        row: dict[str, Any] = {}
        for key, value in zip(columns, fields):
            value = value.strip()
            row[key] = float(value) if _NUMERIC.match(value) else value
        rows.append(row)
    if not rows:
        raise ValueError("FoldX table header found but no data rows beneath it")
    return rows


#: The one term the arm is primarily about, after snake-casing.
INTERACTION_ENERGY_KEY = "interaction_energy"

#: Columns that identify a row rather than measure anything; they are not
#: features and must not reach the regression.
_NON_FEATURE = frozenset({"pdb", "group1", "group2"})


def interaction_features(
    rows: list[dict[str, Any]], group_a: str, group_b: str
) -> dict[str, Any]:
    """Pick the one (group_a, group_b) row and prefix every numeric term.

    AnalyseComplex writes one row per analysed group pair. Selecting it by
    group rather than taking ``rows[0]`` matters as soon as a run analyses more
    than one interface, and costs nothing now.
    """
    wanted = {group_a.strip().upper(), group_b.strip().upper()}
    matches = [
        r
        for r in rows
        if {str(r.get("group1", "")).strip().upper(), str(r.get("group2", "")).strip().upper()}
        == wanted
    ]
    if len(matches) != 1:
        seen = [(r.get("group1"), r.get("group2")) for r in rows]
        raise ValueError(
            f"expected exactly one FoldX row for groups {sorted(wanted)}, found "
            f"{len(matches)} among {seen}"
        )
    row = matches[0]
    if INTERACTION_ENERGY_KEY not in row:
        raise ValueError(
            f"FoldX output has no {INTERACTION_ENERGY_KEY!r} column; columns are "
            f"{sorted(row)}. The header has changed -- re-read it before trusting "
            "any term in this table."
        )
    out = {
        f"{FEATURE_PREFIX}{key}": value
        for key, value in row.items()
        if key not in _NON_FEATURE and isinstance(value, float)
    }
    out[f"{FEATURE_PREFIX}group1"] = row.get("group1")
    out[f"{FEATURE_PREFIX}group2"] = row.get("group2")
    return out


def numeric_feature_columns(row: dict[str, Any]) -> list[str]:
    """The ``foldx_`` columns of ``row`` that are numbers."""
    return sorted(
        k for k, v in row.items() if k.startswith(FEATURE_PREFIX) and isinstance(v, float)
    )


# ---------------------------------------------------------------------------
# One fold, end to end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldxSettings:
    """Everything that changes a number, recorded with every row."""

    repair: bool = False
    #: Which chain group pairs to analyse. ``("C", "A")`` is peptide vs HLA.
    groups: tuple[str, str] | None = None
    timeout_s: int = 3600
    binary: str = ""
    binary_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Provenance stamped onto every row.

        The binary is proprietary and not in the repository, so its SHA-256 is
        the only thing that lets someone with their own copy confirm they ran
        the same build.
        """
        return {
            f"{FEATURE_PREFIX}repair": self.repair,
            f"{FEATURE_PREFIX}binary": Path(self.binary).name if self.binary else "",
            f"{FEATURE_PREFIX}binary_sha256": self.binary_sha256,
        }


def score_fold(
    folder: Path | str,
    work_dir: Path | str,
    foldx: dict[str, str],
    repair: bool = False,
    timeout_s: int = 3600,
    keep_pdb: bool = False,
) -> dict[str, Any]:
    """Convert, (optionally) repair, AnalyseComplex, parse. One fold, one row.

    Every failure becomes a recorded ``status`` on the returned row rather than
    an exception, matching ``structural_features.extract_path``: a FoldX
    failure on one complex is data about that complex, and losing a whole
    container's chunk to it would be worse.
    """
    folder = Path(folder)
    work = Path(work_dir) / folder.name
    started = time.monotonic()
    row: dict[str, Any] = {"complex_id": folder.name, "fold_dir": str(folder)}
    try:
        shutil.rmtree(work, ignore_errors=True)
        (work / "in").mkdir(parents=True)
        (work / "out").mkdir(parents=True)
        # Each fold gets its own CWD with its own link to the FoldX support
        # data. FoldX writes some scratch next to the working directory, and
        # this runner puts dozens of processes on one container: a shared CWD
        # is a race whose symptom would be a wrong number, not a crash.
        for key in ("molecules", "rotabase"):
            source = foldx.get(key)
            if source:
                link = work / Path(source).name
                if not link.exists():
                    link.symlink_to(Path(source))

        prepared = prepare_pdb(folder, work / "in")
        row.update(
            allele=prepared["allele"],
            peptide=prepared["peptide"],
            hla_chain=prepared["hla_chain"],
            peptide_chain=prepared["peptide_chain"],
            beta2m_chain=prepared["beta2m_chain"],
            n_atoms=prepared["conversion"]["n_atoms"],
            max_coord_dev_A=prepared["validation"]["max_coord_dev_A"],
        )
        t_convert = time.monotonic()

        pdb_name = Path(prepared["pdb"]).name
        pdb_dir = str(work / "in")
        out_dir = str(work / "out")

        if repair:
            proc = subprocess.run(
                repair_command(foldx["binary"], pdb_name, pdb_dir, out_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(work),
            )
            repaired = Path(out_dir) / f"{Path(pdb_name).stem}_Repair.pdb"
            if proc.returncode != 0 or not repaired.exists():
                raise RuntimeError(
                    f"RepairPDB rc={proc.returncode}: {(proc.stderr or proc.stdout)[-800:]}"
                )
            shutil.copy2(repaired, Path(pdb_dir) / repaired.name)
            pdb_name = repaired.name
        t_repair = time.monotonic()

        groups = (prepared["peptide_chain"], prepared["hla_chain"])
        proc = subprocess.run(
            analyse_complex_command(foldx["binary"], pdb_name, pdb_dir, out_dir, groups),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(work),
        )
        hits = sorted(Path(out_dir).glob("Interaction_*.fxout"))
        if proc.returncode != 0 or not hits:
            raise RuntimeError(
                f"AnalyseComplex rc={proc.returncode}, {len(hits)} Interaction_*.fxout: "
                f"{(proc.stderr or proc.stdout)[-800:]}"
            )
        if len(hits) != 1:
            raise RuntimeError(f"{len(hits)} Interaction_*.fxout files: {[p.name for p in hits]}")
        features = interaction_features(parse_fxout(hits[0].read_text()), *groups)
        row.update(features)
        row.update(
            FoldxSettings(
                repair=repair,
                binary=foldx["binary"],
                binary_sha256=foldx.get("sha256", ""),
            ).as_dict()
        )
        row.update(
            status="ok",
            convert_s=round(t_convert - started, 3),
            repair_s=round(t_repair - t_convert, 3),
            analyse_s=round(time.monotonic() - t_repair, 3),
            total_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001 -- failures are rows, not crashes
        row["status"] = f"{type(exc).__name__}: {exc}"
        row["total_s"] = round(time.monotonic() - started, 3)
    finally:
        if not keep_pdb:
            shutil.rmtree(work, ignore_errors=True)
    return row


def iter_cohort_folds(root: Path | str) -> Iterator[Path]:
    """Cohort fold directories under ``root``.

    A one-line delegation to :func:`pepstab.structural_features.discover_folds`,
    kept as a named function so the reason is visible at the call site: the
    ``_smoke`` / ``_shards`` / ``_failed`` exclusion has exactly one
    implementation in this repository.
    """
    return discover_folds(Path(root))
