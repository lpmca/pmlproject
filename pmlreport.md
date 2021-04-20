Pratical Machine Learning Course Project
========================================


```r
library(caret)
library(randomForest)
```


```r
train <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```


```r
dim(train)
```

```
## [1] 19622   160
```
We have 160 variables, which we can find that many are largely filled with NA values.

```r
dim(testing)
```

```
## [1]  20 160
```


```r
set.seed(52)
nzv <- nearZeroVar(train)
train<-train[,-nzv]
# Partition the training set into a training and cross validation set
inTrain <- createDataPartition(train$classe,p=0.7,list=FALSE)

xTrain <- train[inTrain,]
xCross <- train[-inTrain,]
```


```r
#We remove the first seven columns becuase these are all filled with information that is irrelevant to our analysis (username, time)
xTrain <- xTrain[,-c(1:7)]
xCross <- xCross[,-c(1:7)]
testing <- testing[,-c(1:7)]

#We remove the columns filled with more than ninety percent NA values
xTrain <- xTrain[,colSums(is.na(xTrain))<nrow(xTrain)*.9]
xTrain$classe <- as.factor(xTrain$classe)
xCross <- xCross[,colSums(is.na(xCross))<nrow(xCross)*.9]
xCross$classe <- as.factor(xCross$classe)
testing <- testing[,colSums(is.na(testing))<nrow(testing)*.9]
```

We perform resampling to better fit our model (prevent over-fitting) which we can do with trainControl

```r
control <- trainControl(method="cv",number=3,verboseIter = F)
```
We are doing random forest for classifying. We find an error rate of 1/6%.

```r
forest <- train(classe~.,data=xTrain,method="rf",trControl=control,tuneLength=5)
forest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 14
## 
##         OOB estimate of  error rate: 0.62%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    3    0    0    0 0.0007680492
## B   13 2639    6    0    0 0.0071482318
## C    0   18 2375    3    0 0.0087646077
## D    0    0   28 2223    1 0.0128774423
## E    0    0    3   10 2512 0.0051485149
```
Here we can do cross validation where we have a similar error rate of 1.5%.

```r
pred_for <- predict(forest$finalModel,newdata=xCross)
confusionMatrix(pred_for,xCross$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    1 1133    4    0    0
##          C    0    3 1022    6    3
##          D    0    0    0  957    3
##          E    1    0    0    1 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9958          
##                  95% CI : (0.9937, 0.9972)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9946          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9947   0.9961   0.9927   0.9945
## Specificity            0.9993   0.9989   0.9975   0.9994   0.9996
## Pos Pred Value         0.9982   0.9956   0.9884   0.9969   0.9981
## Neg Pred Value         0.9995   0.9987   0.9992   0.9986   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1925   0.1737   0.1626   0.1828
## Detection Prevalence   0.2846   0.1934   0.1757   0.1631   0.1832
## Balanced Accuracy      0.9990   0.9968   0.9968   0.9961   0.9970
```
We have found a model that will fairly accurately predict class with about sample error of about 1.5%.
