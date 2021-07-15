library(mlbench)
library(e1071)
library(caret)

install.packages('pROC', dependencies = TRUE)

install.packages('htmltools')

Traindataset <- read.csv("C:/Users/harde/OneDrive/Belgeler/R Projects/tensor/TrainDataset.txt" , header=TRUE, sep=",")


sample <- sample(1:nrow(Traindataset), size=nrow(Traindataset)*0.8)
train <- Traindataset[sample, ]
test <- Traindataset[-sample, ]

x <- train[,-65]
y <- train$TARGET
y <- as.factor(y)
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))


Predict <- predict(model,newdata = test)

cf <- confusionMatrix(table(Predict, test$TARGET))

X <- varImp(model)
plot(X)

plot(model)

library(lares)
mplot_roc(test$TARGET, as.numeric(Predict))



cmatrix <- as.table(cf$table)

fourfoldplot(cmatrix, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

TP <- cmatrix[1]
FP <- cmatrix[2]
FN <- cmatrix[3]
TN <- cmatrix[4]

ACC <- (TP+TN)/(TP+TN+FN+FP)
RECALL <- TP/(TP+FN)
SPECIFICITY <- TN/(TN+FP)
PRECISION <- TP/(TP+FP)
F1 <- 2*PRECISION*RECALL/(PRECISION+RECALL)

temp <- as.numeric(Predict) - as.numeric(test$TARGET)
temp <- temp^2
temp_sum <- sum(temp)

MSE <- temp_sum / length(temp)
RMSE <- sqrt(MSE)

cat("Accuracy:", ACC)
cat("Recall:", RECALL)
cat("Spesificity:", SPECIFICITY)
cat("Precision:", PRECISION)
cat("F1-Score:", F1)
cat("MSE:",MSE)
cat("RMSE:",RMSE)