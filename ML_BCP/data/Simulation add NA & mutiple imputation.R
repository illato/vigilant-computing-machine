# add random missing in Simulated data with signal
#install.packages("missForest")
library(missForest)

# set seed for reproducibility
set.seed(42)

sim_Gail_signal_add_NA<-sim_Gail_signal
colnames(sim_Gail_signal_add_NA)
addna<-prodNA(sim_Gail_signal_add_NA[ , c(4:8)], noNA = 0.2)
sim_Gail_signal_add_NA<-cbind(sim_Gail_signal_add_NA[ , c(1:3)],addna,sim_Gail_signal_add_NA[ , c(9:10)])
#
# Original author just added 20% missing data and is going to change any
# NA's introduced in N_Biop to 0? 
#
# There is already a 0.8 probability that N_Biop will be generated as 0
# in the original "Data Simulation 1 Gail.R" script, implying 0 is
# a valid value, which means there is already implicit imputation in the 
# data that is supposed to be only adding 20% NA. 
#
# Simply put, BRCA example data has 99 (NA) among the values for N_Biop, 
# but none of the data generated here has 99 (NA) for N_Biop.
#
# Additionally, N_Biop of 0 or 99 result in a change of HypPlas to the
# value representing NA/missing (99) below, which doesn't line up with the 
# example data from the BRCA library and I see no clear explanation of 
# the logic behind this modification of the data. 
#
# The example data has 99 for N_Biop with non-zero for HypPlas and vice-versa.
# 
# Run "library(BRCA)" > "data(exampledata)" (then "exampledata", if you are not 
# in an IDE) to see and let me know if any of you see a way where this makes sense.
#
sim_Gail_signal_add_NA$N_Biop[is.na(sim_Gail_signal_add_NA$N_Biop)] <- 0
sim_Gail_signal_add_NA$HypPlas[is.na(sim_Gail_signal_add_NA$HypPlas)] <- 99
sim_Gail_signal_add_NA$AgeMen[is.na(sim_Gail_signal_add_NA$AgeMen)] <- 99
sim_Gail_signal_add_NA$Age1st[is.na(sim_Gail_signal_add_NA$Age1st)] <- 99
sim_Gail_signal_add_NA$N_Rels[is.na(sim_Gail_signal_add_NA$N_Rels)] <- 99

# if N_Biop is 0, change HypPlas to 99 (NA)
sim_Gail_signal_add_NA$HypPlas<-ifelse(sim_Gail_signal_add_NA$N_Biop==0,99,sim_Gail_signal_add_NA$HypPlas)
# if N_Biop is 99 (NA), change HypPlas to 99 (NA)
sim_Gail_signal_add_NA$HypPlas<-ifelse(sim_Gail_signal_add_NA$N_Biop==99,99,sim_Gail_signal_add_NA$HypPlas)

str(sim_Gail_signal_add_NA)
table(sim_Gail_signal_add_NA$HypPlas)

#Multiple imputation

#install.packages("glmnet")
library(glmnet)
#install.packages("mi")
library(mi)
set.seed(1104)
sim_Gail_signal_imputed<-sim_Gail_signal_add_NA
sim_Gail_signal_imputed$N_Biop[sim_Gail_signal_imputed$N_Biop==99] <- NA
sim_Gail_signal_imputed$AgeMen[sim_Gail_signal_imputed$AgeMen==99] <- NA
sim_Gail_signal_imputed$Age1st[sim_Gail_signal_imputed$Age1st==99] <- NA
sim_Gail_signal_imputed$N_Rels[sim_Gail_signal_imputed$N_Rels==99] <- NA
str(sim_Gail_signal_imputed)



## Impute missing with 4 imputed data sets

mdf <- missing_data.frame(sim_Gail_signal_imputed)
str(sim_Gail_signal_imputed)
raw.im <- mi(mdf, n.iter=6, n.chains=4, seed=13423)
raw.complete <- complete(raw.im)
raw.complete$`chain:1`
imputed1<-raw.complete$`chain:1`[1:10]
imputed2<-raw.complete$`chain:2`[1:10]
imputed3<-raw.complete$`chain:3`[1:10]
imputed4<-raw.complete$`chain:4`[1:10]
str(imputed1)
