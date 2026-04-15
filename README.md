# pCO2GalicIA

---

## Overview
This repository contains the R objects and workflows used to estimate deltafCO₂ in the NW Iberian upwelling system. Furthermore, it also contains the data and code to reproduce the main figures of the -currently under review- manuscript.

---

## Key Features
- pCO2GalicIA model is a machine‑learning ensemble approach that integrates three different algorithms: Support Vector Regression (SVR), Extreme Gradient Boost (XGBoost) and Random Forest (RF).
- The file "to_be_named.R" is a reproducible workflow example to estimate deltafCO₂ for a set of input values. The required input variables, besides date and location, are sea surface temperature, sea surface salinity, chlorophyll and upwelling index at 42ºN 10ºW. For chlorophyll and upwelling index the mean values of the last 7-days are also needed.
- Executing the file "code_and_data_to_reproduce_ms.R" will display the figures contained in the associated manuscript. 


---
## Citation
Please, cite (it will be updated after review):
**Air–Sea CO₂ Fluxes in the NW Iberian Upwelling System from pCO₂ Observations and a Machine‑Learning Ensemble**  
Marcos Fontela¹ and Xosé Antonio Padín¹

¹ Instituto de Investigacións Mariñas (IIM‑CSIC), Vigo, Spain  
Contact: mfontela@iim.csic.es
