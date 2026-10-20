# -*- coding: utf-8 -*-
"""
fco2galicia - sea-air delta fCO2 in the NW Iberian upwelling system
(Fontela & Padín, currently under review).

    import fco2galicia as fg
    model = fg.load_model()
    out = model.predict(df)          # predictors + 'date' -> column deltafCO2 (uatm)

Author: Marcos Fontela
"""
from .core import (DEFAULT_MODEL_DIR, OUTPUT_COLUMN, TIME_PREDICTORS, WEEK_PERIOD, FCO2GalicIA,
                   add_time_predictors, load_model)

__version__ = "1.0.0"
__all__ = ["FCO2GalicIA", "load_model", "add_time_predictors", "OUTPUT_COLUMN",
           "TIME_PREDICTORS", "WEEK_PERIOD", "DEFAULT_MODEL_DIR", "__version__"]
