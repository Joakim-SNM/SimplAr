# SimplAr

For the weary ⁴⁰Ar/³⁹Ar geochronologist who wants to extract some nice ages and mess around with different calibrations, and still have time to produce even more sweet data!
If your data is from a mass spectrometer, run it past SimplAr. Create plateaus, isochrons, PDPs, mixture modelling, stack aliquots, combine experiments (maybe some incremental heating and fusion ages?), calibrate your standards through astronomical or joint determination pathways. Create J-vales in 2D or 3D and use the flux covariance for your ages, or recalculate ages on different calibrations. 
From reducing your flux monitor data to a final image and dataset export, this fully independent software lets you take data from any mass spec and process it in bulk. 

Made to be simple, easy to use, and with all the best functions functions from the legacy programs. Hope you like it..

## The tools

In the order they are used (generally):

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
6. **Recalculator and age combiner**: recasts published ages to another calibration
   (any monitor) with the external error of the new calibration, and combines ages published on different calibrations.

The launcher's **Standards** window holds the monitors, calibrations and
decay constants that all tools use, with references. 

The launcher's **About** window describes the methods and equations in full, with
references.

## Installation

SimplAr runs on Windows, macOS and Linux. It needs Python; the packages it uses are
installed at the first start.

1. **Install Python 3.10 or later** from
   [python.org/downloads](https://www.python.org/downloads/).
   - Windows: tick **Add python.exe to PATH** on the first page of the installer.
   - Mac: use the python.org installer. A Python from Homebrew will not let SimplAr
     install its packages.
2. **Download SimplAr**: at the top of this page, click the green **Code** button,
   then **Download ZIP**. Unzip it (on Windows: right-click the ZIP, **Extract All**)
   and put the folder where you want to keep it.
3. **Start it**: in the folder, double-click `launch.bat` (Windows) or
   `launch.command` (Mac). The first start installs the packages SimplAr uses
   (numpy, scipy, pandas, openpyxl, matplotlib, PyQt5), which takes a few minutes.
   After that the launcher opens straight away.

Both systems may warn the first time, because the files come from the internet. On
Windows, click **More info** and **Run anyway** (or **Run**).

On a Mac, if `launch.command` is blocked, open **System Settings → Privacy &
Security**, scroll down to the message about `launch.command`, click **Open Anyway**,
and double-click it again. If it is still blocked (some university and company Macs
do not allow Open Anyway), start it once from Terminal: open the Terminal app, type
`bash` and a space, drag `launch.command` from the Finder into the Terminal window,
and press Return. After that, double-clicking works.

From a terminal, the same is:

    python -m pip install -r requirements.txt
    python launcher.py

(`python3` on macOS and Linux). Each tool also runs on its own, for example
`python 4_plateau_isochron/plateau_isochron.py`.

## Where files are saved

Results and sessions go to a data folder outside the program folder, by default
`SimplAr data` in your Documents folder, with one subfolder per tool
(`1 Data converter` … `6 Recalculator and age combiner`). Every open and save dialog
starts in its tool's subfolder; .jflux files are looked for in `3 Flux regression`.
The launcher shows the data folder and can open it or point the tools at another one.

## Layout

| Path | Contents |
| --- | --- |
| `launcher.py` | launcher and the Methods & Reference document |
| `arar_math.py` | statistics, mixture models, calibration presets and constants shared by all tools |
| `standards.py`, `standards_default.json` | the standards library with its editor, and its published values |
| `export.py` | data export (.xlsx / .csv) and the figure-export dialog |
| `flux_dialog.py` | the disk-flux dialog |
| `data_folders.py` | the data folder and the open / save dialogs |
| `1_data_converter/` … `6_recalculator/` | one folder per tool |
| `tests/` | checks against published values and independent implementations |

## Tests

    python tests/run_all.py

runs every test (the windows open off-screen): calibrations and external errors
(`test_calibration.py`), the combination statistics and mixture models
(`test_statistics.py`), Flux monitors and Flux regression (`test_flux.py`), the figure
exports (`test_figures.py`), and the tools opening, saving and agreeing with each other
(`test_tools.py`), and calibration routes checked independently of the route code (`test_routes.py`). The tests use made-up data
(`tests/synthetic_data.py`) and check the tools against published values and against implementations
written separately from the published equations (`tests/reference_framework.py`).
On GitHub they run on Windows, macOS and Linux for every push
(`.github/workflows/tests.yml`).

## Calibration

All tools refer back a the standards library and they are defined by whether they are  determined by 40K, joint determination or astronomical calibration. If your external errors are usually high because of the decay constant, condider these options. All calculations happen at the 1σ level. 

## Citing

Cite the archived release on Zenodo; GitHub's **Cite this repository** button (from
`CITATION.cff`) gives the reference. The DOI is added here with the first release.

## Licence

GNU General Public License v3.0 or later; see `LICENSE`.
