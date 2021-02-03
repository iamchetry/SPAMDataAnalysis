setwd('../iamchetry/Documents/UB_files/506/hw_6/')

#install.packages('gbm')
#install.packages('kernlab')

library(rpart) 
library(MASS)
library(caret)
library(dplyr)
library(glue)
library(leaps)
library(pROC)
library(randomForest)
library(bootstrap)
library(gbm)
library(kernlab)

#----------------- 1st and 2nd Questions Combined --------------------

load('pima(3).RData')
data_ = pima
data_ = data_[, -8]
attach(data_)
set.seed(10)
t = createDataPartition(class, p=0.7, list = FALSE)
train_ = na.omit(data_[t, ])
test_ = na.omit(data_[-t, ])
y_true = test_$class

#Single Tree Prediction
control_ = rpart.control(minsplit = 10, xval = 5, cp = 0)
tree_ = rpart(class~., data = train_, method = "class", control = control_)

plot(tree_$cptable[,4], main = "Cp for model selection", ylab = "Cp", type='line')

min_cp = which.min(tree_$cptable[,4])
pruned_tree = prune(tree_, cp = tree_$cptable[min_cp,1])

#Feature Importance
plot(pruned_tree$variable.importance, xlab="variable", 
     ylab="Importance", xaxt = "n", pch=20)
axis(1, at=1:length(pruned_tree$variable.importance), 
     labels=names(pruned_tree$variable.importance))

par(mfrow = c(1,2))
plot(pruned_tree, branch = .3, compress=T, main = "Pruned Tree")
text(pruned_tree, cex = .5)

plot(tree_, branch = .3, compress=T, main = "Full Tree")
text(tree_, cex = .5)

my_pred = predict(pruned_tree, newdata = test_, type = "class")

tab_test = table(my_pred, y_true)
conf_test = confusionMatrix(tab_test)

test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

#Bagging
bag_model = randomForest(class~., data = train_, ntree = 500, mtry = 7)
par(mfrow = c(1,2))
varImpPlot(bag_model, main='Feature Importances')
importance(bag_model)

bag_pred = predict(bag_model, newdata = test_, type = "response")
tab_test = table(bag_pred, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

#Random Forest
rf_model = randomForest(class~., data = train_, ntree = 500, mtry = 4)
par(mfrow = c(1,2))
varImpPlot(rf_model, main='Feature Importances')
importance(rf_model)

rf_pred = predict(rf_model, newdata = test_, type = "response")
tab_test = table(rf_pred, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

imp = importance(rf_model)
impvar = rownames(imp)[order(imp[, 1], decreasing=TRUE)]
op = par(mfrow=c(3, 3))
for (i in seq_along(impvar)) {
  partialPlot(rf_model, pred.data=train_, x.var=impvar[i], xlab=impvar[i],
              main=paste("Partial Dependence on", impvar[i]))
}

#LDA
lda_ = lda(class~., data=train_)

lda_preds = predict(lda_, newdata = test_)
tab_test = table(lda_preds$class, y_true)

#--------- Confusion Matrix to determine Accuracy ---------
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4)

#Boosting
train_$class = as.numeric(train_$class)-1
test_$class = as.numeric(test_$class)-1
y_true = as.factor(test_$class)

boost_model = gbm(class~., data = train_, n.trees = 500, shrinkage = 0.01, 
                interaction.depth = 3, distribution = "adaboost")

train_pred = predict(boost_model, newdata = train_, n.trees = 500,
                     type = "response")
boost_pred = predict(boost_model, newdata = test_, n.trees = 500,
                     type = "response")

#Finding optimal threshold to calculate decision boundary
analysis = roc(response=train_$class, predictor=train_pred)
e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
opt_t = subset(e,e[,2]==max(e[,2]))[,1]

test_preds = ifelse(boost_pred >= opt_t, '1', '0')
tab_test = table(test_preds, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


#Log-Reg
train_$class = as.factor(train_$class)
test_$class = as.factor(test_$class)

logreg = glm(class~., data=train_, family='binomial')
train_preds = predict(logreg, newdata = train_, type='response')
test_preds = predict(logreg, newdata = test_, type='response')

#Finding optimal threshold to calculate decision boundary
analysis = roc(response=train_$class, predictor=train_preds)
e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
opt_t = subset(e,e[,2]==max(e[,2]))[,1]

test_preds = ifelse(test_preds >= opt_t, '1', '0')
tab_test = table(test_preds, test_$class)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

#KNN
require(class)
K = c()
E_tr = c()
E_ts = c()

for (k in seq(1, 101, 2))
{
  KNN_train = knn(train_[-c(8)], train_[-c(8)], train_$class, k) # Train Prediction
  KNN_test = knn(train_[-c(8)], test_[-c(8)], train_$class, k) # Test Prediction
  
  train_predicted = as.factor(KNN_train)
  test_predicted = as.factor(KNN_test)
  
  tab_train = table(train_predicted, train_$class)
  tab_test = table(test_predicted, test_$class)
  
  #-------- Confusion Matrix for Accuracy ---------
  conf_train = confusionMatrix(tab_train)
  conf_test = confusionMatrix(tab_test)
  
  K = c(K, k)
  E_tr = c(E_tr, 1 - round(conf_train$overall['Accuracy'], 4))
  E_ts = c(E_ts, 1 - round(conf_test$overall['Accuracy'], 4))
}

par(mfrow = c(2, 1))
plot(K, E_tr, main = 'Train Performance : Choosing different values of k',
     xlab = 'K', ylab = 'Train Error', col='blue', type='line')
plot(K, E_ts, main = 'Test Performance : Choosing different values of k',
     xlab = 'K', ylab = 'Test Error', col='blue', type='line')


#----------------- 3rd Question ------------------

data("spam")
data_ = spam
attach(data_)
set.seed(20)
t = createDataPartition(type, p=0.7, list = FALSE)
train_ = na.omit(data_[t, ])
test_ = na.omit(data_[-t, ])
y_true = test_$type

#Random Forest
M = c()
E = c()

for (m in seq(5, 55, 5)){
  rf_model = randomForest(type~., data = train_, ntree = 500, mtry = m)
  rf_pred = predict(rf_model, newdata = test_, type = "response")
  tab_test = table(rf_pred, y_true)
  conf_test = confusionMatrix(tab_test)
  test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error
  M = c(M, m)
  E = c(E, test_error)
  jpeg(glue('OOB_plot_{m}.jpeg'))
  plot(rf_model$err.rate[,1], main = glue('OOB Error for m={m}'),
       xlab = 'No. of Trees', ylab = 'OOB Error', col='blue')
  dev.off()
}

plot(M, E, main = 'Test Performance : Choosing different values of m',
     xlab = 'No. of Variables chosen', ylab = 'Test Error', col='blue', type='line')









