

# Set working directory to root of repository on your machine -------------

setwd('C:/Users/Bob/Desktop/U/practicum/vigilant-computing-machine/')
#setwd('C:/Users/John/Desktop/U/practicum/vigilant-computing-machine/')
#setwd('C:/Users/Sam/Desktop/U/practicum/vigilant-computing-machine/')


# Run BCRAT/Gail Synthetic Data Generation in Order -----------------------

# uncomment install.packages() above any lines that error out
#
source('./ML_BCP/data/Data Simulation 1 Gail.R')
source('./ML_BCP/code/BCRAT prediction for Simulated data.R')
source('./ML_BCP/data/Simulation add NA & mutiple imputation.R')
source('./ML_BCP/code/BCRAT prediction for Simulated data_w_NA_and_multiple_imputation.R')


# Synthetic BCRAT/Gail Model Data Ready for ML Prediction -----------------