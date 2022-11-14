# CHyPP (Combined Hybrid-Parallel Prediction)
The repository contains the MATLAB code for the 1-Spatial-Dimension version of Combined Hybrid-Parallel Prediction presented in [Wikner et. al. 2020](https://doi.org/10.1063/5.0005541). In addition to the standard CHyPP configuration, this code may also be used to run the parallel machine learning model presented in [Pathak et. al 2018](https://doi.org/10.1103/PhysRevLett.120.024102)(i.e., running CHyPP without a knowledge-based model) and the hybrid model using a single machine learning device presented in [Pathak et. al. 2018 (a)](https://doi.org/10.1063/1.5028373) and used for data assimilation and forecasting in [Wikner et. al. 2021](https://doi.org/10.1063/5.0048050). A 3-Spatial-Dimensional version of CHyPP implemented in Fortran used for terrestrial weather and climate forecasting can be found [here](https://github.com/Arcomano1234/SPEEDY-ML).

## Getting Started
To run CHyPP, you will need to have MATLAB installed. This version of CHyPP was developed using MATLAB 2018b. In addition, to run CHyPP in parallel using multiple cores, you will also need to have MATLAB's Parallel Computing Toolbox installed. If you do not have this toolbox, you can use CHyPP_serial instead of CHyPP.

## Running CHyPP
