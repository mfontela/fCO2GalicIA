# -*- coding: utf-8 -*-
"""
example_basic.py - minimal use of fCO2GalicIA.

Predicts delta fCO2 (uatm) for twelve example observations (2010) and writes
examples/output_example.csv and examples/output_example.png.

    python examples/example_basic.py

Reference: Fontela, M., & Padín, X. A. (currently under review). Air-Sea CO2
Fluxes in the NW Iberian Upwelling System from fCO2 observations and a
machine-learning ensemble.

Author: Marcos Fontela (mfontela@iim.csic.es)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fco2galicia as fg  # noqa: E402

model = fg.load_model()                   # checksums verified
model.self_test()
print(model)

# Input: thetao (degC), so (dimensionless), CHL and CHL_7 (mg m-3),
# UI and UI_7 (m3 s-1 per 100 m of coastline), lat (degN), lon (degE), date (YYYY-MM-DD)
obs = pd.read_csv(os.path.join(HERE, "input_example.csv"))
res = model.predict(obs)                  # adds column deltafCO2 (uatm)

cols = ["date", "lat", "lon", "deltafCO2"]
print(res[cols].round(1).to_string(index=False))
res[cols].to_csv(os.path.join(HERE, "output_example.csv"), index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)
fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
ax.plot(pd.to_datetime(res["date"]), res["deltafCO2"], "-o", color="#08519c", ms=4)
ax.axhline(0, color="0.5", lw=0.7, ls="--")
ax.set_ylabel(r"$\Delta f\mathrm{CO_2}$ ($\mu$atm)")
ax.set_title(r"$f\mathrm{CO_2}$GalicIA - example (2010)", loc="left", fontsize=10)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "output_example.png"))
