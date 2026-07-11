# Compressor Curve Regression Tool

A Streamlit app that turns raw compressor performance data (Speed / Flow / Head / Efficiency /
Power curves exported from a process simulator) into regressed, unit-normalized performance
curves — and calculates whichever parameter (Head, Efficiency, or Power) is missing from the
source data, using gas properties from the workbook.

## What it does

1. **Upload** a multi-sheet `.xlsx` workbook — one sheet per compressor stage.
2. **Detects data blocks** in each sheet automatically:
   - `Speed | Flow | <Head/Efficiency/Power>` triplets, along with their units row
     (`rpm | cfm | ft`, etc.)
   - The `Parameter | Value | Units` operating-conditions table (Diameter, Gas Name,
     Molecular Weight, Pressure, Temperature, Compressibility, Isentropic Exponent)
3. **Converts units** to a consistent standard:
   - Flow → m³/hr
   - Head → m
   - Power → kW
   - Efficiency → %
   - Unrecognized units are left as-is with a visible warning rather than silently guessed.
4. **Regresses** each Speed line for each parameter using your choice of Linear through 5th-order
   polynomial, cubic spline, or an "Auto Best Fit" that picks the highest-R² method.
5. **Calculates the missing parameter** (Head, Efficiency, or Power — whichever the sheet doesn't
   provide) from the other two, using inlet gas density derived from the Operating Conditions
   block:
   - `ρ = P·MW / (Z·R·T)`
   - `ṁ = Q·ρ`
   - `Power = ṁ·g·Head / η`  (rearranged to solve for whichever term is missing)
6. **Exports** a workbook with:
   - Per-stage interpolated curves on a common flow grid, unit-labeled columns
   - `Summary_R2` — regression method and R² per stage/speed/parameter
   - `Operating_Conditions` — consolidated Parameter/Value/Units per stage
   - `Workbook_Overview` — what was found and what was calculated, per stage

## Requirements

- Python 3.9+
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run compressor_curve_app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), and upload a
workbook via the file uploader.

## Expected input format

Each sheet needs, somewhere in the first ~13 columns:

- A `Parameter | Value | Units` table listing at least `Pressure`, `Temperature`,
  `Molecular Weight`, and `Compressibility` (needed for the missing-parameter calculation;
  everything else in that block is carried through but not used in the physics).
- One or more `Speed | <Flow column> | <Head/Efficiency/Power column>` triplets, each with a
  units row directly beneath the header row.

If a sheet is missing the Operating Conditions block, or its Pressure/Temperature/MW/Z aren't
readable as numbers, the app still regresses and exports the curves it can find — it just skips
the missing-parameter calculation for that stage and says so in the UI.

## Known limitations / assumptions to verify

- **Pressure and Temperature are assumed to be absolute inlet (suction) conditions.** If your
  source tool exports gauge pressure or a different station, the calculated missing parameter
  will be off — confirm this against your process simulator's convention.
- **Auto Best Fit currently favors Cubic Spline almost every time.** `CubicSpline` interpolates
  exactly through every point, so it scores a near-perfect R² on training data regardless of
  whether it's the physically sensible curve — this is a known open issue, not yet fixed.
- Unit conversion tables only cover the unit strings seen so far (`cfm`, `ft`, `hp`, `bhp`,
  `kw`, `%`, `m3/hr`, `m`). Other unit labels pass through unconverted with a warning.
- Malformed or non-numeric rows within a data block are silently skipped (`except: pass`) rather
  than reported individually.
- Sheet names are truncated to Excel's 31-character limit on export; two stage names that are
  identical past character 31 will collide.

## File structure

```
.
├── compressor_curve_app.py   # Streamlit app
├── requirements.txt
└── README.md
```
