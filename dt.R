Traindataset <- read.csv("C:/Users/harde/OneDrive/Belgeler/R Projects/tensor/TrainDataset.txt" , header=TRUE, sep=",")
Testdataset <- read.csv("C:/Users/harde/OneDrive/Belgeler/R Projects/tensor/TestDataset.txt" , header=TRUE, sep=",")

library(tree)

set.seed(1815850)

dt <- numeric()

max_acc <- 0

fit <- tree(as.factor(TARGET) ~ .,data = Traindataset)
Predictions <- predict(fit,Testdataset,type="class")
first_predict <- table(predict(fit,Testdataset,type="class"),Testdataset[,"TARGET"])

for (i in 1:10){
  sub <- sample(1:nrow(Traindataset),size=nrow(Traindataset)*0.80)
  fit <- tree(as.factor(TARGET) ~ .,data = Traindataset,subset = sub)
  test_predict <- table(predict(fit,Traindataset[-sub,],type="class"),Traindataset[-sub,"TARGET"])
  dt <- c(dt,sum(diag(test_predict))/sum(test_predict))
}

library(lares)

mplot_roc(Testdataset$TARGET, as.numeric(Predictions))


plot(dt,type = "l",xlab = "Iteration",ylab = "Accuracy",main = "Accuracy changes over 10 iterations")
plot(-dt,type = "l",xlab = "Iteration",ylab = "Error Rate",main = "Error rate changes over 10 iterations")
cat(("Average accuracy is: "),mean(dt))
cat(("Maximum accuracy is: "),max(dt))
cat(("Minimum accuracy is: "),min(dt))
plot(fit,type="uniform")
text(fit)

fourfoldplot(first_predict, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")


TP <- first_predict[1]
FP <- first_predict[2]
FN <- first_predict[3]
TN <- first_predict[4]

ACC <- (TP+TN)/(TP+TN+FN+FP)
RECALL <- TP/(TP+FN)
SPECIFICITY <- TN/(TN+FP)
PRECISION <- TP/(TP+FP)
F1 <- 2*PRECISION*RECALL/(PRECISION+RECALL)

cat("Accuracy:", ACC)
cat("Recall:", RECALL)
cat("Spesificity:", SPECIFICITY)
cat("Precision:", PRECISION)
cat("F1-Score:", F1)