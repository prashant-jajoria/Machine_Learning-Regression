library(psych)
library(ggplot2)
library(reshape2)
library(GGally)
library(tidyverse)
library(leaps)
library(caret)
library(FNN)
library(gridExtra)
library(car)

# Read the dataset
dataset = read.csv('./pmsm_temperature_data_A1_2021.csv')

# Remove the unnecessary columns
dataset = dataset[,c(-10,-11,-12)]

colnames(dataset)

str(dataset)

# check the first few rows
head(dataset)

# check the last few rows
tail(dataset)

# testing set
testing_set = dataset[dataset$profile_id %in% c(72,81), -10]
# training set
training_set = dataset[dataset$profile_id !=72 & dataset$profile_id !=81 , -10]

round(describe(training_set),3)

# reshanping for plotting purpose
m1 <- melt(as.data.frame(training_set))

ggplot(m1,aes(x = variable,y = value)) +
    facet_wrap(~variable, scales="free") +
    geom_boxplot() +
    scale_y_continuous(labels=function (n) {format(n, scientific=FALSE)})

outliers_ambient = boxplot.stats(training_set$ambient)$out
outliers_u_d = boxplot.stats(training_set$u_d)$out
outliers_torque = boxplot.stats(training_set$torque)$out
outliers_i_q = boxplot.stats(training_set$i_q)$out

outliers <- c()
for (name in colnames(training_set)){
    outliers <- c(outliers, 
                  (which(training_set[,name] %in% boxplot(training_set[,name],plot=F)$out)) )
}
total_outliers = length(unique(outliers))

##outlier values
paste( "There are" , length(outliers_ambient), "outliers in feature Ambient" )
paste( "There are" , length(outliers_u_d), "outliers in feature u_d" )
paste( "There are" , length(outliers_torque), "outliers in feature torque" )
paste( "There are" , length(outliers_i_q), "outliers in feature i_q" )

paste("In total there are", total_outliers, 
      "outliers in the dataset, which forms", 
      (total_outliers / dim(training_set)[1]) * 100,
      "% of the training dataset"
     )

library(Hmisc)
hist.data.frame(training_set)

g1 = ggplot(aes(x=i_d), data=training_set) +
      geom_histogram(bins=20)

g2 = ggplot(aes(x=log(i_d+max(i_d)+10,10)), data=training_set) +
      geom_histogram(bins=20)

g3 = ggplot(aes(x=(i_d)**2), data=training_set) +
      geom_histogram(bins=20)

g4 = ggplot(aes(x=(i_d +max(i_d)+10)**(1/3)), data=training_set) +
      geom_histogram(bins=20)


gridExtra::grid.arrange(g1,g2,g3,g4, nrow=2,ncol=2)

g1 = ggplot(aes(x=coolant), data=training_set) +
      geom_histogram(bins=20)

g2 = ggplot(aes(x=log(coolant+max(i_d)+10,10)), data=training_set) +
      geom_histogram(bins=20)

g3 = ggplot(aes(x=(coolant)**2), data=training_set) +
      geom_histogram(bins=20)

g4 = ggplot(aes(x=(coolant +max(coolant)+10)**(1/3)), data=training_set) +
      geom_histogram(bins=20)



gridExtra::grid.arrange(g1,g2,g3,g4, nrow=2,ncol=2)

g1 = ggplot(aes(x=motor_speed), data=training_set) +
      geom_histogram(bins=20)

g2 = ggplot(aes(x=log(motor_speed+max(i_d)+10,10)), data=training_set) +
      geom_histogram(bins=20)

g3 = ggplot(aes(x=(motor_speed)**2), data=training_set) +
      geom_histogram(bins=20)

g4 = ggplot(aes(x=(motor_speed +max(motor_speed)+10)**(1/3)), data=training_set) +
      geom_histogram(bins=20)



gridExtra::grid.arrange(g1,g2,g3,g4, nrow=2,ncol=2)

ggpairs(training_set)

Model.Accuracy <- function(predicted, target) {
    rss <- 0
    tss <- 0
    se <- 0
    target.mean <- mean(target)
    mse <- 0
    
    for (i in 1:length(predicted)) {
        rss <- rss + (predicted[i]-target[i])^2
        tss <- tss + (target[i]-target.mean)^2
        
        se <- se + (predicted[i]-target[i])^2
    }
    rmse <- sqrt(se/length(predicted))
    rsquared <- 1 - rss/tss
    
    mse = mean((predicted - target)^2)
    
    return(list(rsquared=rsquared, rss=rss, tss=tss, rmse=rmse, mse=mse))
}

rmse <- function(predicted, target) {
    se <- 0
    for (i in 1:length(predicted)) {
        se <- se + (predicted[i]-target[i])^2
    }
    return (sqrt(se/length(predicted)))
}

lm.model.base <- lm(pm ~ ., data=training_set)
summary(lm.model.base)

i_s = c(sqrt(training_set['i_d']*training_set['i_d'] + training_set['i_q']*training_set['i_q'])) 
i_s_test = c(sqrt(testing_set['i_d']*testing_set['i_d'] + testing_set['i_q']*testing_set['i_q'])) 

u_s = c(sqrt(training_set['u_d']*training_set['u_d'] + training_set['u_q']*training_set['u_q'])) 
u_s_test = c(sqrt(testing_set['u_d']*testing_set['u_d'] + testing_set['u_q']*testing_set['u_q'])) 

# adding the new features to training and test set
training_set_new = data.frame(cbind(training_set,i_s,u_s))
test_set_new = data.frame(cbind(testing_set,i_s_test,u_s_test))

colnames(training_set_new) =  c('ambient' ,'coolant' ,'u_d', 'u_q' ,'motor_speed' ,'torque', 'i_d' ,'i_q' ,'pm','i_s','u_s')
colnames(test_set_new) =  c('ambient' ,'coolant' ,'u_d', 'u_q' ,'motor_speed' ,'torque', 'i_d' ,'i_q' ,'pm','i_s','u_s')

# computing active power
active_power = c(1.5* (training_set_new[,3]*training_set_new[,7] + training_set_new[,4]*training_set_new[,8]) )
training_set_new = data.frame(cbind(training_set_new, active_power))

active_power = c(1.5* (test_set_new[,3]*test_set_new[,7] + test_set_new[,4]*test_set_new[,8]) ) 
test_set_new = data.frame(cbind(test_set_new, active_power))

# computing reactive power
reactive_power = c(1.5 * (training_set_new[,4]*training_set_new[,8] - training_set_new[,3]*training_set_new[,7])  ) 
training_set_new = data.frame(cbind(training_set_new, reactive_power))

reactive_power = c(1.5 * (test_set_new[,4]*test_set_new[,8] - test_set_new[,3]*test_set_new[,7]) ) 
test_set_new = data.frame(cbind(test_set_new, reactive_power))

lm.model.no_interactions <- lm(pm ~ ., data=training_set_new)
summary(lm.model.no_interactions)

lm.model.interactions <- lm(pm ~ . + torque*i_q + u_d*i_q + u_q*motor_speed + motor_speed*i_d + u_d*torque , data=training_set_new)
summary(lm.model.interactions)

lm.model.exhaustive <- regsubsets(pm ~ . + torque*i_q + u_d*i_q + u_q*motor_speed + motor_speed*i_d + u_d*torque , data = training_set_new)
reg.summary <- summary(lm.model.exhaustive)
reg.summary

par(mfrow = c(2, 2))
plot(reg.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2, pch = 20)
plot(reg.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20)
plot(reg.summary$adjr2, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, pch = 20)
plot(reg.summary$rss, xlab = "Number of variables", ylab = "RSS", type = "l")
mtext("Plots of C_p, BIC, adjusted R^2 and RSS for forward stepwise selection", side = 3, line = -2, outer = TRUE)

lm.model.forward <- regsubsets(pm ~ . + torque*i_q + u_d*i_q + u_q*motor_speed + motor_speed*i_d + u_d*torque , data = training_set_new)
reg.summary <- summary(lm.model.forward)
reg.summary

par(mfrow = c(2, 2))
plot(reg.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2, pch = 20)
plot(reg.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20)
plot(reg.summary$adjr2, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, pch = 20)
plot(reg.summary$rss, xlab = "Number of variables", ylab = "RSS", type = "l")
mtext("Plots of C_p, BIC, adjusted R^2 and RSS for forward stepwise selection", side = 3, line = -2, outer = TRUE)

lm.model.backward <- regsubsets(pm ~ . + torque*i_q + u_d*i_q + u_q*motor_speed + motor_speed*i_d + u_d*torque , data = training_set_new)
reg.summary <- summary(lm.model.backward)
reg.summary

par(mfrow = c(2, 2))
plot(reg.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2, pch = 20)
plot(reg.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20)
plot(reg.summary$adjr2, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, pch = 20)
plot(reg.summary$rss, xlab = "Number of variables", ylab = "RSS", type = "l")
mtext("Plots of C_p, BIC, adjusted R^2 and RSS for forward stepwise selection", side = 3, line = -2, outer = TRUE)

lm.model.reduced <- lm(pm ~ . + u_q*motor_speed + motor_speed*i_d, data=training_set[,-c(6,8,10,11,12)])
summary(lm.model.reduced)

par(mfrow=c(2,2))
plot(lm.model.reduced)

outlierTest(lm.model.reduced, cutoff=0.05, digits = 1)

influencePlot(lm.model.reduced, scale=5, id.method="noteworthy", main="Influence Plot", sub="Circle size is proportial to Cook's Distance" )

lm.model.reduced.no_outliers <- lm(pm ~ . + u_q*motor_speed + motor_speed*i_d, data=training_set[ -c(4162,4172,4192,4201,4221,1471) ,-c(6,8,10,11,12)])
summary(lm.model.reduced.no_outliers)

par(mfrow=c(2,2))
plot(lm.model.reduced.no_outliers)

# predictions for different models
y_pred_train_1 = predict(lm.model.reduced.no_outliers, training_set_new[-c(4162,4172,4192,4201,4221,1471),])
y_pred_train_2 = predict(lm.model.reduced, training_set_new[-c(4162,4172,4192,4201,4221,1471),])
y_pred_train_3 = predict(lm.model.base, training_set[-c(4162,4172,4192,4201,4221,1471),])
y_pred_train_4 = predict(lm.model.no_interactions, training_set_new[-c(4162,4172,4192,4201,4221,1471),])
y_pred_train_5 = predict(lm.model.interactions, training_set_new[-c(4162,4172,4192,4201,4221,1471),])

# metrics
paste("Chosen model")
Model.Accuracy(y_pred_train_1,training_set_new[-c(4162,4172,4192,4201,4221,1471),]$pm)
paste("Model with outliers")
Model.Accuracy(y_pred_train_2,training_set_new[-c(4162,4172,4192,4201,4221,1471),]$pm)
paste("BaseLine model")
Model.Accuracy(y_pred_train_3,training_set[-c(4162,4172,4192,4201,4221,1471),]$pm)
paste("Model without Intearction terms")
Model.Accuracy(y_pred_train_4,training_set_new[-c(4162,4172,4192,4201,4221,1471),]$pm)
paste("Model with interaction terms")
Model.Accuracy(y_pred_train_5,training_set_new[-c(4162,4172,4192,4201,4221,1471),]$pm)

# predictions for different models
y_pred_train_1 = predict(lm.model.reduced.no_outliers, training_set_new[c(4162,4172,4192,4201,4221,1471),])
y_pred_train_2 = predict(lm.model.reduced, training_set_new[c(4162,4172,4192,4201,4221,1471),])
y_pred_train_3 = predict(lm.model.base, training_set[c(4162,4172,4192,4201,4221,1471),])
y_pred_train_4 = predict(lm.model.no_interactions, training_set_new[c(4162,4172,4192,4201,4221,1471),])
y_pred_train_5 = predict(lm.model.interactions, training_set_new[c(4162,4172,4192,4201,4221,1471),])

# metrics
paste("Chosen model")
Model.Accuracy(y_pred_train_1,training_set_new[c(4162,4172,4192,4201,4221,1471),]$pm)
paste("Model with outliers")
Model.Accuracy(y_pred_train_2,training_set_new[c(4162,4172,4192,4201,4221,1471),]$pm)
paste("BaseLine model")
Model.Accuracy(y_pred_train_3,training_set[c(4162,4172,4192,4201,4221,1471),]$pm)
paste("Model without Intearction terms")
Model.Accuracy(y_pred_train_4,training_set_new[c(4162,4172,4192,4201,4221,1471),]$pm)
paste("Model with interaction terms")
Model.Accuracy(y_pred_train_5,training_set_new[c(4162,4172,4192,4201,4221,1471),]$pm)


training_set_new[c(4162,4172,4192,4201,4221,1471),]

y_pred_1 = predict(lm.model.reduced.no_outliers, test_set_new)
y_pred_2 = predict(lm.model.reduced, test_set_new)
y_pred_3 = predict(lm.model.base, testing_set)
y_pred_4 = predict(lm.model.no_interactions, test_set_new)
y_pred_5 = predict(lm.model.interactions, test_set_new)

paste("Chosen model")
Model.Accuracy(y_pred_1,test_set_new$pm)
paste("Model with outliers")
Model.Accuracy(y_pred_2,test_set_new$pm)
paste("BaseLine model")
Model.Accuracy(y_pred_3,testing_set$pm)
paste("Model without Intearction terms")
Model.Accuracy(y_pred_4,test_set_new$pm)
paste("Model with interaction terms")
Model.Accuracy(y_pred_5,test_set_new$pm)

min_max_processor1 <- preProcess(training_set_new, method=c("range"))
min_max_training_set <- predict(min_max_processor1,training_set_new)

min_max_processor2 <- preProcess(test_set_new, method=c("range"))
min_max_testing_set <- predict(min_max_processor2,test_set_new)

set.seed(123)
grid = expand.grid(k = seq(1,20)) # dataframe of different k values

train.control.knn <- trainControl(method = "cv", 
                                  number = 10,
                                  search="grid",)

# Train the model
knn_cv_grid <- train(pm ~. , 
                data = min_max_training_set, 
                method = "knn",
                trControl = train.control.knn,
                tuneGrid = grid)

knn_cv_grid

plot(knn_cv_grid)

prediction.knn.train = knn.reg(
                            train = min_max_training_set[,-9], 
                            y = min_max_training_set[,9], 
                            test = min_max_training_set[,-9], 
                            k = 4
                        )

df.prediction.knn.train = data.frame(observed = min_max_training_set$pm, 
                       predicted = prediction.knn.train$pred )

df.prediction.knn.train %>%
  ggplot(aes(observed, predicted)) +
    geom_hline(yintercept = 0) +
    geom_point() +
    stat_smooth(method = "loess") +
    theme_minimal()+
    ggtitle("Predictions on Training Set")

prediction.knn.test = knn.reg(
                            train = min_max_training_set[,-9], 
                            y = min_max_training_set[,9], 
                            test = min_max_testing_set[,-9], 
                            k = 4
                        )

df.prediction.knn.test = data.frame(observed = min_max_testing_set$pm, 
                       predicted = prediction.knn.test$pred )

df.prediction.knn.test %>%
  ggplot(aes(observed, predicted)) +
    geom_hline(yintercept = 0) +
    geom_point() +
    stat_smooth(method = "loess") +
    theme_minimal()+
    ggtitle("Predictions on Testing Set")

df.error.knn = data.frame(k = double(), 
                           train.rmse = double(),
                           test.rmse = double()
                           )

for (k in 1:200){
    prediction.knn.train = knn.reg(
                            train = min_max_training_set[,-9], 
                            y = min_max_training_set[,9], 
                            test = min_max_training_set[,-9], 
                            k = k
                        )
    
    prediction.knn.test = knn.reg(
                            train = min_max_training_set[,-9], 
                            y = min_max_training_set[,9], 
                            test = min_max_testing_set[,-9], 
                            k = k
                        )
    
    rmse.train = rmse(prediction.knn.train$pred, min_max_training_set$pm)
    rmse.test = rmse(prediction.knn.test$pred, min_max_testing_set$pm)
    
    df.error.knn = rbind(df.error.knn, data.frame(k = k, train.rmse = rmse.train, test.rmse = rmse.test) )
}

prediction_errors_long <- melt(df.error.knn, id="k")

ggplot(data=prediction_errors_long,
       aes(x=k, y=value, colour=variable)) +
       geom_line()

y_pred_train_1 = predict(lm.model.reduced.no_outliers, training_set_new[-c(4162,4172,4192,4201,4221,1471),])
paste("Linear regression model")
Model.Accuracy(y_pred_train_1,training_set_new[-c(4162,4172,4192,4201,4221,1471),]$pm)

paste("Metrics of KNN k=4")
Model.Accuracy(prediction.knn.train$pred, min_max_training_set$pm)

y_pred_1 = predict(lm.model.reduced.no_outliers, test_set_new)
paste("Linear regression model")
Model.Accuracy(y_pred_1,test_set_new$pm)

paste("Metrics of KNN k=4")
Model.Accuracy(prediction.knn.test$pred, min_max_testing_set$pm)

summary(lm.model.reduced.no_outliers)

cor(training_set[-c(4162, 4172, 4192, 4201, 4221, 1471), 
        -c(6, 8, 10, 11, 12)])
