# Multiple-Penalized-State-Space-Models

... Under Construction ...

## test_ssm_M5T200N2P1
This script solves the small-scale simulation with N=2 hidden processes and P=1 autoregressive matrix order using the Matlab Econometric Toolbox ssm.estimate (Maximum likelihood parameter estimation of state-space models). Change the observation noise level in lines 13-15 to test different signal-to-noise ratio.

## test_ssm_M5T240N3P3
This script solves the small-scale simulation with N=3 hidden processes and P=3 autoregressive matrix order using the Matlab Econometric Toolbox ssm.estimate (Maximum likelihood parameter estimation of state-space models). Change the observation noise level in lines 13-15 to test different signal-to-noise ratio.

## test_bssm_M5T200N2P1
This script solves the small-scale simulation with N=2 hidden processes and P=1 autoregressive matrix order using the Matlab Econometric Toolbox bssm.estimate (Bayesian parameter estimation of state-space models). Change the observation noise level in lines 13-15 to test different signal-to-noise ratio.

## test_bssm_M5T240N3P3
This script solves the small-scale simulation with N=3 hidden processes and P=3 autoregressive matrix order using the Matlab Econometric Toolbox bssm.estimate (Bayesian parameter estimation of state-space models). Change the observation noise level in lines 13-15 to test different signal-to-noise ratio.

## test_ssm_M5T200N2P1_MC
This script replicates the simulation in the script "test_ssm_M5T200N2P1" using Monte Carlo simulations, as represented by the number of simulated epochs (Nepoch). The simulated data is read in lines 8-10 from a mat file to guarantee that the same data is used by each method in the comparison analysis.

## test_ssm_M5T240N3P3_MC
This script replicates the simulation in the script "test_ssm_M5T240N3P3" using Monte Carlo simulations, as represented by the number of simulated epochs (Nepoch). The simulated data is read in lines 8-10 from a mat file to guarantee that the same data is used by each method in the comparison analysis.

## test_bssm_M5T200N2P1_MC
This script replicates the simulation in the script "test_bssm_M5T200N2P1" using Monte Carlo simulations, as represented by the number of simulated epochs (Nepoch). The simulated data is read in lines 8-10 from a mat file to guarantee that the same data is used by each method in the comparison analysis.

## test_bssm_M5T240N3P3_MC
This script replicates the simulation in the script "test_bssm_M5T240N3P3" using Monte Carlo simulations, as represented by the number of simulated epochs (Nepoch). The simulated data is read in lines 8-10 from a mat file to guarantee that the same data is used by each method in the comparison analysis.
