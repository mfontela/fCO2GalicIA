# =============================================================================
# example_basic.R - minimal use of fCO2GalicIA from R (through reticulate)
#
#   setwd("path/to/fCO2GalicIA")
#   source("examples/example_basic.R")
#
# Reference: Fontela, M., & Padín, X. A. (currently under review). Air-Sea CO2
# Fluxes in the NW Iberian Upwelling System from fCO2 observations and a
# machine-learning ensemble.
#
# Author: Marcos Fontela (mfontela@iim.csic.es)
#
# Ushey, K., Allaire, J. J., & Tang, Y. (2025). reticulate: Interface to
#   'Python'. R package. https://CRAN.R-project.org/package=reticulate
# =============================================================================
library(reticulate)

# Python with numpy and pandas: built automatically by reticulate >= 1.41,
# or set RETICULATE_PYTHON to an interpreter of your own.
if (Sys.getenv("RETICULATE_PYTHON") == "" && packageVersion("reticulate") >= "1.41.0")
  py_require(c("numpy>=1.24", "pandas>=2.0"), python_version = ">=3.10")

fg    <- import_from_path("fco2galicia", path = normalizePath("."))
model <- fg$load_model()
invisible(model$self_test())
print(model)

obs  <- read.csv(file.path("examples", "input_example.csv"))
vars <- setdiff(unlist(model$predictors), unlist(fg$TIME_PREDICTORS))
obs$deltafCO2 <- unlist(model$predict_columns(lapply(obs[vars], as.numeric),
                                              as.character(as.Date(obs$date))))
print(obs[, c("date", "lat", "lon", "deltafCO2")], digits = 3)
