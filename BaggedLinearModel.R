#Install and load libraries
install.packages(c("mlbench","MASS","parallel","doParallel","Metrics"))

library(mlbench)
library(MASS)
library(parallel)
library(doParallel)
library(Metrics)

#Load sample dataset
data(BostonHousing)
head(BostonHousing)

Index = sample(1:nrow(BostonHousing), 50)

Test = BostonHousing[Index,]
Training = BostonHousing[-Index,]
Predictors.Columns = 1:13
Target.Variable = 14


LM_Bagged = function(X){
  #Bootstrapping
  DATA = Training[sample(1:nrow(Training),nrow(Training),replace=T),]
  
  #Random Selection of Predictors
  PRED = sample(Predictors.Columns, round(length(Predictors.Columns)/3,0))
  
  #Fitting the model
  formula = as.formula(paste0("medv~",paste0(names(DATA[,PRED]),collapse="+")))
  null.lm <- lm(medv~1, data=DATA)
  full.lm <- lm(formula , data=DATA)
  
  LM.STEP.AIC1 = stepAIC(null.lm, scope=list(lowr=null.lm, upper=full.lm), directiom="both")
  
  PRED_LM = predict(LM.STEP.AIC1, newdata=Test)
  
  matrix(PRED_LM, nrow=1)
}



#Perform the analysis in parallel
cluster <- makeCluster(detectCores() - 1) #Leave 1 core for OS
registerDoParallel(cluster)
clusterEvalQ(cl=cluster, library(MASS))
clusterExport(cl=cluster, c("Training", "Test", "Predictors.Columns", "Target.Variable"))


RES_LM = parSapply(cluster, X=1:1000, FUN=LM_Bagged)

stopCluster(cluster)
registerDoSEQ()


PRED_LM = apply(RES_LM, 1, median)
ERR_LM = apply(RES_LM, 1, sd)

RMSE.LM = rmse(RES_LM, Test[,Target.Variable])
MAE.LM = mae(RES_LM, Test[,Target.Variable])