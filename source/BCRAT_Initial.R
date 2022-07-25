

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

sim_Gail_Random
sim_Gail_signal
sim_Gail_signal_add_NA
#sim_Gail_signal_imputed
#
# 'sim_Gail_signal_imputed' is actually modified a surprising amount in the
# 'code/BCRAT prediction for Simulated data_w_NA_and_multiple_imputation.R'
# that does prediction after the data is generated. The result is 'imputed1'
#
# Included logic mentioned above when aggregated/parameterized data generation
# See: './source/BCRAT_Synthetic_Data_Generator.R'
imputed1


# Write Synthetic Data to CSV ---------------------------------------------

# orange3-conformal is in Python

write.csv(sim_Gail_Random, 
          file = './ML_BCP/data/synthetic/random.csv', 
          row.names = FALSE)
write.csv(sim_Gail_signal, 
          file = './ML_BCP/data/synthetic/signal.csv', 
          row.names = FALSE)
write.csv(sim_Gail_signal_add_NA, 
          file = './ML_BCP/data/synthetic/missing.csv', 
          row.names = FALSE)
write.csv(imputed1, 
          file = './ML_BCP/data/synthetic/imputed.csv', 
          row.names = FALSE)
