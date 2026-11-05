# SimplAr

Desktop tools for ⁴⁰Ar/³⁹Ar geochronology, from the laboratory's data export to
plateau, isochron and combined ages. Every age carries three uncertainty tiers:
analytical; internal (plus J); external (plus the monitor age and decay constant).
Shared J and calibration errors are treated as correlated, so they do not average
down when ages are combined.

## The tools

In the order they are normally used:

1. **Data converter**: reads an ArArCALC summary export or a Mass Spec raw export
   (.xlsx) and writes `.arar` session files for the other tools.
2. **Flux monitors**: reduces the Fish Canyon monitor analyses from each irradiation
   position to one weighted-mean R (and J), with outlier rejection and REML
   overdispersion.
3. **Flux regression**: fits J (or R) across the irradiation disk, with Monte Carlo
   uncertainties, and writes a `.jflux` file.
4. **Plateau & isochron**: step-heating age spectra, plateau and isochron ages, and
   the combination of several aliquots (stack mode).
5. **Age distribution**: probability density plots, pooled ages of many analyses or
   samples, and mixture and minimum-age models.

The launcher's **About** window describes the methods and equations in full, with
references.

## Installation

Python 3.10 or later (Windows, macOS or Linux), then:

    pip install -r requirements.txt

## Running

    python launcher.py

or double-click `launch.bat` (Windows) or `launch.command` (macOS). Each tool also
runs on its own, for example `python 4_plateau_isochron/plateau_isochron.py`.

## Where files are saved

Results and sessions go to a data folder outside the program folder, by default
`SimplAr data` in your Documents folder, with one subfolder per tool
(`1 Data converter` … `5 Age distribution`). Every open and save dialog starts in its
tool's subfolder; .jflux files are looked for in `3 Flux regression`. The launcher shows
the data folder and can open it or point the tools at another one. Figure-export presets
and the rendered equations of the About window are kept in the user's settings and
cache folders, so the program folder never changes when you use it.

## Layout

| Path | Contents |
| --- | --- |
| `launcher.py`, `about_text.py` | launcher and the Methods & Reference document |
| `arar_math.py` | statistics, calibration presets and constants shared by all tools |
| `data_export.py`, `figure_export.py`, `flux_dialog.py`, `mixture_model.py` | shared data export, figure export, disk-flux dialog and mixture models |
| `data_folders.py` | the data folder and the open / save dialogs |
| `1_data_converter/` … `5_age_distribution/` | one folder per tool |
| `tests/` | checks against published values and independent implementations |

## Tests

    python tests/run_all.py

runs every test (the windows open off-screen). The tests use made-up data
(`tests/synthetic_data.py`) and check the tools against published values (the
calibration papers, the metafor random-effects example) and against implementations
written separately from the published equations (`tests/reference_framework.py`).
On GitHub they run on Windows, macOS and Linux for every push
(`.github/workflows/tests.yml`).

## Calibration

All tools open on the Phillips et al. (2022) Fish Canyon age with the Min et al. (2000)
decay constant. Changing `DEFAULT_FC_NAME` in `arar_math.py` moves every tool to
another preset. Uncertainties are held at 1σ internally; 1σ or 2σ is chosen at display
and export.

## Citing

A DOI will be added with the first archived release.
