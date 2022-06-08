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

# Number of Biopsy 
N_Biop<- sample(c(0,1,2,3,4,5,6),size=1200, replace=TRUE, prob=c(0.8,0.1,0.02,0.07,0.005,0.004,0.001))#Need change in HypPlas Size
table(N_Biop)
count_Biop<-length(N_Biop[N_Biop!=0])
count_Biop  #242
# N_Biop
# 0   1   2   3   4   5   6 
# 958 111  30  87   9   3   2 

# Hyperplasia
HypPlas <- ifelse(N_Biop!=0,(sample(c(0,1),size=count_Biop,replace=TRUE,prob=c(0.8,0.2))),99) #99=unk or not applicable
table(HypPlas)
# HypPlas
# 0   1  99 
# 191  51 958 

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
Case_signalYN<-ifelse(Case_signal>17.91104, 1, 0)
table(Case_Random)
# Case_Random
# 0   1 
# 586 614 
table(Case_signalYN)
# Case_signalYN
# 0   1 
# 622 578 


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

