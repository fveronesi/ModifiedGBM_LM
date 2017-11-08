#Install and load libraries
install.packages(c("mlbench","dismo","gbm","parallel","doParallel","Metrics"))

library(mlbench)
library(dismo)
library(gbm)
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

BoostedRegressionForest = function(X){
  
  #Bootstrapping
  DATA = Training[sample(1:nrow(Training),nrow(Training),replace=T),]
  
  #Random Selection of Predictors
  PRED = sample(Predictors.Columns, round(length(Predictors.Columns)/3,0))
  
  #Fitting the model
  GBM.mod = gbm.step(data=DATA, gbm.x=PRED, gbm.y=Target.Variable, family="gaussian")
  
  #Please see help(gbm.step) for setting the hyperparameters
  
  PRED_GBM = predict.gbm(GBM.mod, newdata=Test, n.trees=GBM.mod$gbm.call$best.trees, type="response")
  
  matrix(PRED_GBM, nrow=1)
}



#Perform the analysis in parallel
cluster <- makeCluster(detectCores() - 1) #Leave 1 core for OS
registerDoParallel(cluster)
clusterEvalQ(cl=cluster, library(dismo))
clusterEvalQ(cl=cluster, library(gbm))
clusterExport(cl=cluster, c("Training", "Test", "Predictors.Columns", "Target.Variable"))

#nTree allows to choose the number of parallel trees to fit
nTree = 100

RES_GBM = parSapply(cluster, X=1:nTree, FUN=BoostedRegressionForest)

stopCluster(cluster)
registerDoSEQ()


PRED_GBM = apply(RES_GBM, 1, median)
ERR_GBM = apply(RES_GBM, 1, sd)

RMSE.GBM = rmse(PRED_GBM, Test[,Target.Variable])
MAE.GBM = mae(PRED_GBM, Test[,Target.Variable])