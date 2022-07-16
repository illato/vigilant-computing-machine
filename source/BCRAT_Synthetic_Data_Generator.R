# The original author didn't set.seed()--AFAIK..
# So, none of the commented outputs would be reproducible
#
# set seed for reproducibility
set.seed(42)

# The author must have been on Linux/Unix
#
# On Windows, this resolves to C:\Users\<you>\Documents\Desktop\ML_Gail 
#
##setwd("~/Desktop/ML_Gail")
#
# I don't see where setting the directory is relevant anyway

NumSubj <- 1200

ID<-sample(1200, 1200, replace=F)

#age 
T1 <- as.integer(rnorm(NumSubj, 52,5))
table(T1)
# 35  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  69 
# 1   2   2   5   7  11  14  22  28  38  46  85  65  85  88  92 104  83  87  77  65  45  37  45  25  17   5   9   2   6   2 
T2 <- c(rep(90, 1200))

# Why did the original author make the below
# probability of having had 3 biopsies greater than the 
# probability of having had 2 biopsies? 
#
# What does the author mean "Need change in HypPlas Size"?
# HypPlas is based on N_Biop, but beyond that it is not yet clear what was meant

# Number of Biopsy 
N_Biop<- sample(c(0,1,2,3,4,5,6),size=1200, replace=TRUE, prob=c(0.8,0.1,0.02,0.07,0.005,0.004,0.001))#Need change in HypPlas Size
table(N_Biop)
count_Biop<-length(N_Biop[N_Biop!=0])
count_Biop  #242
# N_Biop
# 0   1   2   3   4   5   6 
# 958 111  30  87   9   3   2 

# Hyperplasia
HypPlas <- ifelse(N_Biop!=0,
                  (sample(c(0,1),size=count_Biop,replace=TRUE,prob=c(0.8,0.2))),
                  99) #99=unk or not applicable
table(HypPlas)
# HypPlas
# 0    1  99 
# 198 44 958 

#Age of menarche
AgeMen<- as.integer(rnorm(NumSubj, 13,1))
table(AgeMen)
# AgeMen
# 9  10  11  12  13  14  15  16 
# 2  27 171 429 387 157  26   1 

#Age first live birth 
Age1st<- c(ifelse(runif(NumSubj)<.8 , as.integer(rnorm(NumSubj,25,1.5)),98)) #98=nulliparous
table(Age1st)
# Age1st
# 20  21  22  23  24  25  26  27  28  29  98 
# 3  13  70 146 225 246 167  71  23   4 232 

#Number of Rekatives have breast cancer
N_Rels<- sample(c(0,1,2,3,4,5,6),size=1200, replace=TRUE, prob=c(0.6,0.2,0.1,0.08,0.01,0.005,0.005))
table(N_Rels)
# N_Rels
# 0   1   2   3   4   5   6 
# 710 230 134 102  13   5   6 

#Race
Race <- sample(c(1:7),size=1200, replace=TRUE, prob=c(0.5,0.2,0.2,0.08,0.01,0.005,0.005))
table(Race)
# Race
# 1   2   3   4   5   6   7 
# 589 234 260  94  17   4   2 

# Why did the original author have this here??
#
# Doesn't appear related to Gail/BCRAT, Tyrer-Cuzick/IBIS, or BOADICEA?
#
#BRCA1 <- sample(c(0,1,"NA"),size=1200, replace=TRUE, prob=c(0.2,0.1,0.7))
#BRCA2 <- sample(c(0,1,"NA"),size=1200, replace=TRUE, prob=c(0.2,0.05,0.75))
#Weight <- as.integer(rnorm(NumSubj, 80,10))
#Hight <- as.integer(rnorm(NumSubj, 173,10))

# having cancer
Case_Random<-sample(c(0,1),size=1200, replace=TRUE, prob=c(0.5,0.5))
Case_signal<-c(0.3*T1+0.2*N_Biop+0.1*N_Rels+0.1*AgeMen+0.3*rnorm(20,3))
median(Case_signal) # 17.86993
Case_signalYN<-ifelse(Case_signal>17.86993, 1, 0)
table(Case_Random)
# Case_Random
# 0   1 
# 586 614 
table(Case_signalYN)
# Case_signalYN
# 0   1 
# 601 599 


# Data (putting all components together)

sim_Gail_Random<- cbind(ID,T1,T2,N_Biop,HypPlas,AgeMen,Age1st,N_Rels,Race,Case_Random)
sim_Gail_signal<- cbind(ID,T1,T2,N_Biop,HypPlas,AgeMen,Age1st,N_Rels,Race,Case_signalYN)

#Assign the column names

colnames(sim_Gail_Random) <- c(
  "ID",
  "T1","T2",
  "N_Biop", "HypPlas", "AgeMen",
  "Age1st", "N_Rels", "Race", "Case_Random")
sim_Gail_Random<- data.frame(sim_Gail_Random)
summary(sim_Gail_Random)
str(sim_Gail_Random)

colnames(sim_Gail_signal) <- c(
  "ID",
  "T1","T2",
  "N_Biop", "HypPlas", "AgeMen",
  "Age1st", "N_Rels", "Race", "Case_signalYN")
sim_Gail_signal<- data.frame(sim_Gail_signal)
summary(sim_Gail_signal)
str(sim_Gail_signal)

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
# The example data has NA for N_Biop with non-zero/NA for HypPlas and vice-versa.
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

