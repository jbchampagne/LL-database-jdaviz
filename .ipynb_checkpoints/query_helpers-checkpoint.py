"""
query_helpers.py
----------------
User-friendly functions for querying and appending to emission_lines.ecsv,
so nobody needs to write SQL or raw astropy masking to use the database.

Examples
--------
>>> from query_helpers import load_db, get_lines, get_wavelength_quantity, append_file

>>> db = load_db()
>>> iron_lines = get_lines(db, name_contains="Fe")
>>> nir_lines = get_lines(db, wave_min=1.0, wave_max=2.5, unit="um")
>>> co_lines = get_lines(db, source="CO.csv")

>>> waves = get_wavelength_quantity(iron_lines)   # returns an astropy Quantity

>>> append_file("my_new_linelist.csv", rest_wavelength_col="Rest Value",
...             wavelength_unit="um", extra_cols=["notes"])
"""

import json
import yaml
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from astropy import units as u

DB_FILE = "emission_lines.ecsv"
SCHEMA_FILE = "schema.yaml"


def load_db(db_file=DB_FILE):
    """Load the consolidated database as an astropy Table."""
    return Table.read(db_file, format="ascii.ecsv")


def get_lines(db, name_contains=None, source=None, wave_min=None, wave_max=None, unit=None):
    """
    Return a filtered copy of the database. All arguments are optional and
    combine as AND conditions.

    Parameters
    ----------
    db : astropy.table.Table
        The loaded database (from load_db()).
    name_contains : str, optional
        Case-insensitive substring match on line_name, e.g. "Fe" or "CO".
    source : str, optional
        Restrict to rows from one original source file, e.g. "CO.csv".
    wave_min, wave_max : float, optional
        Wavelength bounds. Must be given in the same `unit` as below.
    unit : str, optional
        Unit that wave_min/wave_max are expressed in (e.g. "um", "Angstrom").
        Required if wave_min or wave_max is given, since the database mixes
        units across rows.

    Returns
    -------
    astropy.table.Table
        Filtered subset (same columns as db).
    """
    mask = np.ones(len(db), dtype=bool)

    if name_contains is not None:
        mask &= np.char.find(np.char.lower(db["line_name"].astype(str)), name_contains.lower()) >= 0

    if source is not None:
        mask &= db["source_list"] == source

    if wave_min is not None or wave_max is not None:
        if unit is None:
            raise ValueError("Please specify `unit` (e.g. 'um' or 'Angstrom') when filtering by wavelength.")
        # convert each row's wavelength into the requested unit for comparison
        converted = np.array([
            (row["rest_wavelength"] * u.Unit(row["wavelength_unit"])).to_value(u.Unit(unit))
            for row in db
        ])
        if wave_min is not None:
            mask &= converted >= wave_min
        if wave_max is not None:
            mask &= converted <= wave_max

    return db[mask]


def get_wavelength_quantity(db_subset):
    """
    Convert the rest_wavelength + wavelength_unit columns of a (sub)table
    into a single astropy Quantity array, handling mixed units row-by-row.
    Returned in the unit of the first row if units are mixed; otherwise
    returns them natively.
    """
    if len(db_subset) == 0:
        return u.Quantity([])
    if len(set(db_subset["wavelength_unit"])) == 1:
        unit = u.Unit(db_subset["wavelength_unit"][0])
        return db_subset["rest_wavelength"] * unit
    # mixed units: convert everything to the first row's unit
    target_unit = u.Unit(db_subset["wavelength_unit"][0])
    return u.Quantity([
        (row["rest_wavelength"] * u.Unit(row["wavelength_unit"])).to(target_unit)
        for row in db_subset
    ])


def get_extra_info(row):
    """Parse a row's extra_info JSON string into a plain Python dict."""
    return json.loads(row["extra_info"]) if row["extra_info"] else {}


def append_file(filename, rest_wavelength_col, wavelength_unit, extra_cols=None,
                 db_file=DB_FILE, schema_file=SCHEMA_FILE):
    """
    Append a new source CSV file to the existing database (and record it
    in schema.yaml so future rebuilds stay reproducible).

    Parameters
    ----------
    filename : str
        Path to the new raw CSV. Must have a "Line Name" column plus a
        wavelength column.
    rest_wavelength_col : str
        Name of the wavelength column in the new file (e.g. "Rest Value").
    wavelength_unit : str
        Astropy-parsable unit string for that column, e.g. "um" or "Angstrom".
    extra_cols : list of str, optional
        Any additional columns to preserve as per-row metadata.
    """
    extra_cols = extra_cols or []

    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    unit = u.Unit(wavelength_unit)  # validates the unit string

    names, waves, extras, sources = [], [], [], []
    for _, row in df.iterrows():
        name = row["Line Name"]
        wave = row[rest_wavelength_col]
        if pd.isna(name) or pd.isna(wave):
            continue
        extra = {c: row[c] for c in extra_cols if c in df.columns and not pd.isna(row[c])}
        names.append(str(name).strip())
        waves.append(float(wave))
        extras.append(json.dumps(extra) if extra else "{}")
        sources.append(filename)

    new_tbl = Table()
    new_tbl["line_name"] = names
    new_tbl["rest_wavelength"] = np.array(waves)
    new_tbl["wavelength_unit"] = [wavelength_unit] * len(names)
    new_tbl["source_list"] = sources
    new_tbl["extra_info"] = extras

    existing = Table.read(db_file, format="ascii.ecsv")
    merged = vstack([existing, new_tbl], metadata_conflicts="silent")
    merged.meta = existing.meta
    merged.write(db_file, format="ascii.ecsv", overwrite=True)

    # record this file in schema.yaml for reproducibility of future full rebuilds
    with open(schema_file, "r") as f:
        schema = yaml.safe_load(f)
    schema["extra_info_columns_by_file"][filename] = {
        "rest_wavelength_col": rest_wavelength_col,
        "wavelength_unit": wavelength_unit,
        "extra_cols": extra_cols,
    }
    with open(schema_file, "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)

    print(f"Appended {len(new_tbl)} rows from {filename}. Database now has {len(merged)} rows total.")
    return merged
