# *f*CO₂GalicIA

**Machine-Learning Ensemble for Air-Sea ΔfCO₂ Estimation in the NW Iberian Upwelling System**

This model generates estimates of the sea–air difference in CO₂ fugacity
(ΔfCO₂ = fCO₂,sw − fCO₂,atm, µatm) at the ocean surface of the NW Iberian upwelling system
(41.5–43.0°N, 10.5–8.0°W) from user-supplied inputs of location, date, sea surface
temperature, sea surface salinity, surface chlorophyll-a concentration, and the coastal
Upwelling Index. A positive ΔfCO₂ indicates that the ocean is a CO₂ source to the
atmosphere; a negative value indicates uptake.

The model is the machine-learning ensemble described in:

> Fontela, M., & Padín, X. A. (currently under review). Air–Sea CO₂ Fluxes in the NW Iberian
> Upwelling System from fCO₂ observations and a machine-learning ensemble.

Model design and performance are documented in the article; please cite it when using the model.

## How to use

```bash
pip install -r requirements.txt
python examples/example_basic.py
```

```python
import pandas as pd
import fco2galicia as fg

model = fg.load_model()
obs = pd.read_csv("examples/input_example.csv")   # predictors + date
out = model.predict(obs)                          # adds deltafCO2 (µatm)
```

From R, through reticulate: `source("examples/example_basic.R")`.

## Input

| Column | Units | Variable |
|---|---|---|
| `thetao` | °C | sea surface temperature |
| `so` | dimensionless | sea surface salinity |
| `CHL` | mg m⁻³ | surface chlorophyll-a |
| `CHL_7` | mg m⁻³ | 7-day trailing median of `CHL` |
| `UI` | m³ s⁻¹ km coastline⁻¹ | Upwelling Index at 42°N 10°W |
| `UI_7` | m³ s⁻¹ km coastline⁻¹ | 7-day trailing median of `UI` |
| `lat`, `lon` | °N, °E | position (longitude negative west) |
| `date` | YYYY-MM-DD | year and ISO week are derived from it |

Rows with a missing input return NaN. Inputs outside the valid ranges listed in
`model/fCO2GalicIA/manifest.json` raise an extrapolation warning.

## Repository structure

```
fCO2GalicIA/
├── README.md                     this file
├── LICENSE.txt                   MIT licence
├── CITATION.cff                  citation metadata
├── pyproject.toml                package definition (pip install .)
├── requirements.txt              dependencies (numpy, pandas; matplotlib optional)
├── fco2galicia/                  prediction package (NumPy + pandas)
│   ├── __init__.py               public API: load_model, add_time_predictors
│   └── core.py                   ensemble evaluators, input checks, R bridge
├── model/
│   └── fCO2GalicIA/
│       ├── manifest.json         predictors, units, valid ranges, checksums, reference predictions
│       ├── random_forest.npz     Random Forest
│       ├── svr.npz               Support Vector Regression (RBF kernel)
│       └── xgboost.npz           XGBoost
├── examples/
│   ├── example_basic.py          demonstration script (Python)
│   ├── example_basic.R           demonstration script (R, through reticulate)
│   ├── input_example.csv         example input data
│   ├── output_example.csv        example output (generated)
│   └── output_example.png        example figure (generated)
└── tests/
    └── test_fco2galicia.py       pytest suite
```

## Frequently asked questions

**Q. Can I use the model with data from outside the 41.5–43.0°N, 10.5–8.0°W domain?**  
A. The model will return a numerical value, but it will constitute extrapolation, and the
package issues a warning. The same applies to any input outside the valid ranges listed in
`model/fCO2GalicIA/manifest.json`, including years after 2021. Results outside the training
domain should be treated with caution and validated against independent observations
whenever possible.

**Q. Why are 7-day medians needed for chlorophyll and UI?**  
A. Biological carbon uptake and the physical response of the carbonate system to
upwelling pulses operate on timescales of days to a week. The lagged predictors
(`CHL_7` and `UI_7`) were found to substantially improve model performance, particularly
during transitional seasons.

**Q. Can I use instantaneous daily satellite images instead of weekly composites?**  
A. Yes. The model is agnostic to the temporal resolution of the input data, as long
as the units are consistent with those listed above. `CHL_7` and `UI_7` must still be
7-day trailing medians of `CHL` and `UI`.

**Q. How is the ensemble prediction computed?**  
A. `deltafCO2` is the unweighted arithmetic mean of the three individual model predictions
(Random Forest, Support Vector Regression and XGBoost). A row with any missing input returns
NaN.

## License

MIT ([LICENSE.txt](LICENSE.txt)).

## Contact

Marcos Fontela (mfontela@iim.csic.es)
Instituto de Investigacións Mariñas (IIM-CSIC), Vigo, Spain.
