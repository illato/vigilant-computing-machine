# #Gail absolut risk 
# #library("BCRA", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
# #
# # Original author doesn't make it apparent what version of "BCRA" is used
# #
# # The commit was on 21 MAR 2019, which would imply BCRA v2.1 <https://cran.r-project.org/src/contrib/Archive/BCRA/>
# #
# # The latest version is BCRA v2.1.1, which implies no breaking changes (otherwise you would bump to v2.2)
# #
# # If we find incompatibilities, build the source from <"https://cran.r-project.org/src/contrib/Archive/BCRA/BCRA_2.1.tar.gz">
# # or try the following:
# #require(devtools)
# #install_version("BCRA", version = "2.1", repos = "http://cran.us.r-project.org")
# #
# # Trying v2.1.1 first, per corresponds to the Reference Manual <https://cran.r-project.org/web/packages/BCRA/BCRA.pdf>
# 
# #install.packages("BCRA")
# library(BCRA)
# 
# # set seed for reproducibility
set.seed(42)

# #Simulated Gail no signal
# Abrisk_Random<-risk.summary(sim_Gail_Random)
# Abrisk_Random<-as.data.frame(Abrisk_Random)
# str(Abrisk_Random)
# Abrisk_Random$AbsRisk<-as.numeric(as.character(Abrisk_Random$AbsRisk))
# mean(Abrisk_Random$AbsRisk)
# Predition_Gail_Random <- ifelse(Abrisk_Random$AbsRisk<median(Abrisk_Random$AbsRisk), 0, 1)
# table(Predition_Gail_Random)
# sim_Gail_Random_prediction<-cbind(sim_Gail_Random,Predition_Gail_Random)
# 
# #Simulated Gail atifical signal
# Abrisk_signal<-risk.summary(sim_Gail_signal)
# Abrisk_signal<-as.data.frame(Abrisk_signal)
# str(Abrisk_signal)
# Abrisk_signal$AbsRisk<-as.numeric(as.character(Abrisk_signal$AbsRisk))
# Predition_Gail_signal <- ifelse(Abrisk_signal$AbsRisk<median(Abrisk_signal$AbsRisk), 0, 1)
# table(Predition_Gail_signal)
# sim_Gail_signal_prediction<-cbind(sim_Gail_signal,Predition_Gail_signal)
# 
# #Calculate AUC 
# #install.packages("crossval")
# library("crossval")
# sim_GRan_AUC = crossval::confusionMatrix(sim_Gail_Random_prediction$Case_Random, sim_Gail_Random_prediction$Predition_Gail_Random, negative=0)
# 
# # FP  TP  TN  FN 
# # 299 303 287 311
# sim_GSig_AUC = confusionMatrix(sim_Gail_signal_prediction$Case_signalYN, sim_Gail_signal_prediction$Predition_Gail_signal, negative=0)
# # FP  TP  TN  FN 
# # 347 255 275 323
# diagnosticErrors(sim_GRan_AUC)
# # acc        sens        spec         ppv         npv         lor 
# # 0.49166667  0.49348534  0.48976109  0.50332226  0.47993311 -0.06702146 
# diagnosticErrors(sim_GSig_AUC)
# # acc       sens       spec        ppv        npv        lor 
# # 0.4458333  0.4474124  0.4442596  0.4451827  0.4464883 -0.4350237 
# 
# #install.packages("cvAUC")
# library(cvAUC)
# 
# AUC(sim_Gail_Random_prediction$Predition_Gail_Random, sim_Gail_Random_prediction$Case_Random, label.ordering = NULL)
# AUC(sim_Gail_signal_prediction$Predition_Gail_signal, sim_Gail_signal_prediction$Case_signalYN, label.ordering = NULL)

# Commenting below for now per:
#
# Original author committed the code that generates the 
# ..._add_NA variables after committing the code that uses them
#
#Gail absolut risk 
Abrisk_signal_add_NA<-risk.summary(sim_Gail_signal_add_NA)
Abrisk_signal_add_NA<-as.data.frame(Abrisk_signal_add_NA)
str(Abrisk_signal_add_NA)
Abrisk_signal_add_NA$AbsRisk<-as.numeric(as.character(Abrisk_signal_add_NA$AbsRisk))
Predition_Gail_signal_add_NA <- ifelse(Abrisk_signal_add_NA$AbsRisk<median(Abrisk_signal_add_NA$AbsRisk), 0, 1)
table(Predition_Gail_signal_add_NA)
sim_Gail_signal_addNA_prediction<-cbind(sim_Gail_signal_add_NA,Predition_Gail_signal_add_NA)

#Calculate AUC
sim_GSig_addNA_AUC = confusionMatrix(sim_Gail_signal_addNA_prediction$Case_signalYN, sim_Gail_signal_addNA_prediction$Predition_Gail_signal_add_NA, negative=0)
# FP  TP  TN  FN
#350 250 251 349
diagnosticErrors(sim_GSig_addNA_AUC)
#      acc       sens       spec        ppv        npv        lor
#0.4175000  0.4173623  0.4176373  0.4166667  0.4183333 -0.6660912

AUC(sim_Gail_signal_addNA_prediction$Predition_Gail_signal_add_NA, sim_Gail_signal_addNA_prediction$Case_signalYN, label.ordering = NULL)

#imputed1 gail risk
imputed1$HypPlas<-as.numeric(as.character(imputed1$HypPlas))
imputed1$Case_signalYN<-as.numeric(as.character(imputed1$Case_signalYN))
imputed1$Case_signalYN<-as.numeric(imputed1$Case_signalYN)
imputed1$AgeMen<-round(imputed1$AgeMen)
imputed1$Age1st<-round(imputed1$Age1st)
imputed1$N_Rels<-round(imputed1$N_Rels)
imputed1$N_Rels<-ifelse(imputed1$N_Rels<0,0,imputed1$N_Rels)
imputed1$Age1st<-ifelse(imputed1$Age1st<0,98,imputed1$Age1st)
imputed1$Age1st<-ifelse(imputed1$Age1st>45,40,imputed1$Age1st)
imputed1$Age1st<-ifelse(imputed1$Age1st<16,18,imputed1$Age1st)
imputed1$Age1st<-ifelse(imputed1$Age1st>imputed1$T1,T1,imputed1$Age1st)
table(imputed1$HypPlas)
str(imputed1)
Gail.imputed1 <- risk.summary(imputed1)
Gail.imputed1<-as.data.frame(Gail.imputed1)
str(Gail.imputed1)
Gail.imputed1$AbsRisk<-as.numeric(as.character(Gail.imputed1$AbsRisk))
str(Gail.imputed1)
Predition_Gail_imput1 <- ifelse(Gail.imputed1$AbsRisk<median(Gail.imputed1$AbsRisk,na.rm=T), 0, 1)
table(Predition_Gail_imput1)
sim_Gail_imput1_prediction<-cbind(imputed1,Predition_Gail_imput1)

#Calculate AUC
sim_Gail_imput1_prediction_AUC = confusionMatrix(sim_Gail_imput1_prediction$Case_signalYN, sim_Gail_imput1_prediction$Predition_Gail_imput1, negative=0)
# FP  TP  TN  FN
# 327 273 274 326
diagnosticErrors(sim_Gail_imput1_prediction_AUC)
#       acc       sens       spec        ppv        npv        lor
# 0.4558333  0.4557596  0.4559068  0.4550000  0.4566667 -0.3542577

AUC(sim_Gail_imput1_prediction$Predition_Gail_imput1, sim_Gail_imput1_prediction$Case_signalYN, label.ordering = NULL)
