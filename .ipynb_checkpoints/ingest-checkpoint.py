"""
ingest.py
---------
Reads all raw emission-line CSV files in this directory, validates them
against schema.yaml, and merges them into a single consolidated
emission_lines.ecsv database.

Usage:
    python ingest.py

To add a NEW source file later:
    1. Drop the new .csv into this directory.
    2. Add an entry for it under `extra_info_columns_by_file` in schema.yaml
       (rest_wavelength_col, wavelength_unit, and any extra columns to keep).
    3. Re-run `python ingest.py`. It will re-read *all* source files and
       rebuild emission_lines.ecsv from scratch (safe/idempotent), OR see
       `append_new_file()` below to add just one file to an existing database.
"""

import json
import yaml
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from astropy import units as u

SCHEMA_FILE = "schema.yaml"
OUTPUT_FILE = "emission_lines.ecsv"


def load_schema(schema_file=SCHEMA_FILE):
    with open(schema_file, "r") as f:
        return yaml.safe_load(f)


def _clean_value(v):
    """Turn NaN / empty-ish values into None so they drop out of the JSON blob,
    and strip stray non-ASCII garbage from string fields."""
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.strip()
        v = v.encode("ascii", errors="ignore").decode("ascii")
        if v == "":
            return None
    return v


def read_one_file(filename, file_schema):
    """Read a single raw CSV into a normalized astropy Table with core
    columns + a JSON 'extra_info' column."""
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    wave_col = file_schema["rest_wavelength_col"]
    unit_str = file_schema["wavelength_unit"]
    extra_cols = file_schema.get("extra_cols", [])

    # validate the unit is a real astropy unit
    unit = u.Unit(unit_str)

    # validate wavelength column exists and sanity-check the range
    if wave_col not in df.columns:
        raise ValueError(f"{filename}: expected wavelength column '{wave_col}' not found")

    names, waves, extras, sources = [], [], [], []
    for _, row in df.iterrows():
        name = _clean_value(row["Line Name"])
        wave = _clean_value(row[wave_col])
        if name is None or wave is None:
            continue  # skip malformed rows rather than silently corrupting the DB

        extra = {}
        for col in extra_cols:
            if col in df.columns:
                val = _clean_value(row[col])
                if val is not None:
                    extra[col] = val

        names.append(str(name))
        waves.append(float(wave))
        extras.append(json.dumps(extra) if extra else "{}")
        sources.append(filename)

    tbl = Table()
    tbl["line_name"] = names
    tbl["rest_wavelength"] = np.array(waves)
    tbl["wavelength_unit"] = [unit_str] * len(names)
    tbl["source_list"] = sources
    tbl["extra_info"] = extras

    tbl["rest_wavelength"].unit = None  # keep raw float; unit is tracked in wavelength_unit col
    tbl.meta["source_file_units"] = {filename: unit_str}
    return tbl


def sanity_check(tbl, schema):
    bounds = schema.get("wavelength_sanity_bounds", {})
    for unit_str, (lo, hi) in bounds.items():
        mask = tbl["wavelength_unit"] == unit_str
        bad = mask & ((tbl["rest_wavelength"] < lo) | (tbl["rest_wavelength"] > hi))
        if bad.any():
            bad_rows = tbl[bad]
            print(f"WARNING: {bad.sum()} row(s) with unit '{unit_str}' fall outside "
                  f"expected range [{lo}, {hi}]:")
            print(bad_rows["line_name", "rest_wavelength", "source_list"])


def build_database(schema_file=SCHEMA_FILE, output_file=OUTPUT_FILE):
    schema = load_schema(schema_file)
    file_configs = schema["extra_info_columns_by_file"]

    tables = []
    for filename, file_schema in file_configs.items():
        print(f"Reading {filename} ...")
        tbl = read_one_file(filename, file_schema)
        print(f"  -> {len(tbl)} valid rows")
        tables.append(tbl)

    master = vstack(tables, metadata_conflicts="silent")
    master.meta = {
        "description": "Consolidated emission line database",
        "core_columns": schema["core_columns"],
        "note": "Use wavelength_unit alongside rest_wavelength to reconstruct "
                "a Quantity, e.g. row['rest_wavelength'] * u.Unit(row['wavelength_unit']). "
                "extra_info is a JSON string of any per-source-file metadata.",
    }

    sanity_check(master, schema)

    master.write(output_file, format="ascii.ecsv", overwrite=True)
    print(f"\nWrote {len(master)} total rows to {output_file}")
    return master


if __name__ == "__main__":
    build_database()
