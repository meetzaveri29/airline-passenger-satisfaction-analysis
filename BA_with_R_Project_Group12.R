# Clear the workspace
rm(list=ls())
cat("\014")
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(reshape2)
library(ggplot2)
library(ROCit)

# load in the data file into data.frame
airline <- read.csv("airlinepassengerdata.csv", stringsAsFactors = FALSE)
summary(airline)

# Removing missing values
airline <- na.omit(airline)

summary(airline)

# Changing variables into factors
airline$Gender <- as.factor(airline$Gender)
airline$Customer.Type <- as.factor(airline$Customer.Type)
airline$Type.of.Travel <- as.factor(airline$Type.of.Travel)
airline$Class <- as.factor(airline$Class)
#columns_to_convert <- 8:21
# Convert columns 4 through 10 into factors
#for (col in columns_to_convert) {
#  airline[, col] <- factor(airline[, col])
#}
airline$satisfaction <- as.factor(airline$satisfaction)

#Below are some of the explanatory data analysis using Different Visualization Techniques
#### histograms for the distribution of numerical variables such as Age, Flight Distance, Departure Delay in Minutes, and Arrival Delay in Minutes

# Histogram for Age
ggplot(airline, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency") +
  theme_minimal()

# Histogram for Flight Distance
ggplot(airline, aes(x = Flight.Distance)) +
  geom_histogram(binwidth = 100, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Flight Distance", x = "Flight Distance", y = "Frequency") +
  theme_minimal()

# Function to calculate percentages
calculate_percentages <- function(x) {
  return(prop.table(table(x)) * 100)
}

### comparing the percentages of satisfied and neutral or dissatisfied passengers within categories like gender, type of travel and class.

# Create a summary dataframe with percentages
summary_data <- with(airline, prop.table(table(Gender, satisfaction), margin = 1) * 100)

# Convert summary dataframe to long format
summary_data_long <- reshape2::melt(summary_data)

# Bar chart for Gender with percentages
ggplot(summary_data_long, aes(x = Gender, y = value, fill = satisfaction)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = paste0(round(value, 1), "%")), position = position_stack(vjust = 0.5)) +
  labs(title = "Distribution of Passenger Satisfaction by Gender",
       x = "Gender",
       y = "Percentage") +
  scale_fill_manual(values = c("satisfied" = "green", "neutral or dissatisfied" = "red"))

summary_data_travel_type <- with(airline, prop.table(table(Type.of.Travel, satisfaction), margin = 1) * 100)
# Convert summary dataframe to long format for Type of Travel
summary_data_travel_type_long <- reshape2::melt(summary_data_travel_type)
# Bar chart for Type of Travel with percentages
ggplot(summary_data_travel_type_long, aes(x = Type.of.Travel, y = value, fill = satisfaction)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = paste0(round(value, 1), "%")), position = position_stack(vjust = 0.5)) +
  labs(title = "Distribution of Passenger Satisfaction by Type of Travel",
       x = "Type of Travel",
       y = "Percentage") +
  scale_fill_manual(values = c("satisfied" = "green", "neutral or dissatisfied" = "red"))

# Create a summary dataframe with percentages for Class
summary_data_class <- with(airline, prop.table(table(Class, satisfaction), margin = 1) * 100)
# Convert summary dataframe to long format for Class
summary_data_class_long <- reshape2::melt(summary_data_class)
# Bar chart for Class with percentages
ggplot(summary_data_class_long, aes(x = Class, y = value, fill = satisfaction)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = paste0(round(value, 1), "%")), position = position_stack(vjust = 0.5)) +
  labs(title = "Distribution of Passenger Satisfaction by Class",
       x = "Class",
       y = "Percentage") +
  scale_fill_manual(values = c("satisfied" = "green", "neutral or dissatisfied" = "red"))

#####Use scatter plots to visualize the relationship between two numerical variables, such as Flight distance vs. Departure Delay or Arrival Delay.

# Scatter plot for Flight Distance vs. Departure Delay
ggplot(airline, aes(x = Flight.Distance, y = Departure.Delay.in.Minutes)) +
  geom_point() +
  labs(title = "Flight Distance vs. Departure Delay",
       x = "Flight Distance",
       y = "Departure Delay (Minutes)")

# Scatter plot for Flight Distance vs. Arrival Delay
ggplot(airline, aes(x = Flight.Distance, y = Arrival.Delay.in.Minutes)) +
  geom_point() +
  labs(title = "Flight Distance vs. Arrival Delay",
       x = "Flight Distance",
       y = "Arrival Delay (Minutes)")

###### box plots to visualize the distribution of numerical variables across different levels of categorical variables such as Class.

# Create box plots for numerical variables across different levels of categorical variables
boxplot_data <- airline %>%
  select(Class, Inflight.wifi.service, Food.and.drink, Inflight.entertainment, Seat.comfort, satisfaction)

# Melt the data for box plot
melted_data <- melt(boxplot_data, id.vars = c("Class", "satisfaction"))

# Plot the box plots
ggplot(melted_data, aes(x = variable, y = value, fill = satisfaction)) +
  geom_boxplot() +
  facet_grid(Class ~ ., scales = "free") +
  labs(x = "Variable", y = "Value", fill = "Satisfaction") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Classification Analysis 
# Use the train index set to split the dataset
#  satisfaction.train for building the model
#  satisfaction.test for testing the model
airline_new <- airline[,-1]
set.seed(1000) # for reproducible results
train <- sample(1:nrow(airline_new), nrow(airline_new)*(2/3)) # replace=FALSE by default

satisfaction.train <- airline_new[train,]
satisfaction.test <- airline_new[-train,]

#Goal 1
fit <- rpart(satisfaction ~ ., # formula, all predictors will be considered in splitting
             data=satisfaction.train, # dataframe used
             method="class",  # treat churn as a categorical variable, default
             control=rpart.control(xval=0, minsplit=1000), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
             parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

fit

rpart.plot(fit, type = 1, extra = 4, main="Classification Passenger Satisfaction")

# extract the vector of predicted class for each observation in satisfaction.train
satisfaction.pred <- predict(fit, satisfaction.train, type="class")
# extract the actual class of each observation in chur.train
satisfaction.actual <- satisfaction.train$satisfaction

confusion.matrix <- table(satisfaction.pred, satisfaction.actual)  
confusion.matrix
addmargins(confusion.matrix)
pt <- prop.table(confusion.matrix)  
pt
#accuracy
pt[1,1] + pt[2,2]
# On Training model the accuracy comes out to be 88.46%

# Testing the model on test data
# extract the predicted class of each observation in satisfaction.test
satisfaction.pred <- predict(fit, satisfaction.test, type="class")
# extract the actual class of each observation in satisfaction.test
satisfaction.actual <- satisfaction.test$satisfaction

# now build the "confusion matrix"
# which is the contingency matrix of predicted vs actual
# use this order: predicted then actual
confusion.matrix <- table(satisfaction.pred, satisfaction.actual)  
confusion.matrix
tp <- confusion.matrix[2,2];tp
tn <- confusion.matrix[1,1];tn
fp <- confusion.matrix[2,1];fp
fn <- confusion.matrix[1,2];fn

accuracy <- (tp + tn)/(tp + tn + fp + fn);accuracy
# TPR = Recall = Sensitivity
TPR <- tp/(fn+tp);TPR
# TNR = Specificity
TNR <- tn/(fp+tn);TNR
# FPR
FPR <- fp/(fp+tn);FPR
# FNR
FNR <- fn/(fn+tp);FNR

#The first node of Decision tree is the Online.boarding factor , meaning it has the highest information gain fllowing Inflight.wifi.service and Type of travel which further drives satisfaction of passengers
#Accuracy is 88.38% for the test data

#Goal 2
#Customer Satisfaction based on Time Delays, Flight distance, Checkin, Boarding Service and Flight Class

#logistic
logit.reg <- glm(satisfaction ~ Online.boarding + Checkin.service + Departure.Delay.in.Minutes + 
                   Arrival.Delay.in.Minutes + Departure.Arrival.time.convenient + Flight.Distance + Class, 
                 data = satisfaction.train, family = "binomial") 
summary(logit.reg)

# use predict() with type = "response" to compute predicted probabilities. 
logitPredict <- predict(logit.reg, satisfaction.test, type = "response")
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)

# evaluate classifier on test
actual <- satisfaction.test$satisfaction
predict <- logitPredictClass
cm <- table(predict, actual)
cm
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
(tp + tn)/(tp + tn + fp + fn)
# TPR = Recall = Sensitivity
tp/(fn+tp)
# TNR = Specificity
tn/(fp+tn)
# FPR
fp/(fp+tn)
# FNR
fn/(fn+tp)
# Pvalue is less than 0.05 for all variables except Departure Delay in minutes, so the rest of the factors are significantly affecting the passenger satisfaction except Departure Delay in minutes 
# Accuracy of the model 79.9%

# create ROC curve
roc_logit <- rocit(score = logitPredict, class = actual) 

# check AUC, Cutoff, TPR, FPR(=1-Specificity)
result_logit <- data.frame(cbind(AUC=roc_logit$AUC, Cutoff=roc_logit$Cutoff, 
                                 TPR=roc_logit$TPR, FPR=roc_logit$FPR))
head(result_logit)
tail(result_logit)

# find the optimal point (Youden Index point)
result_logit$diff <- result_logit$TPR - result_logit$FPR
bestcutoff <- result_logit[which.max(result_logit[, c("diff")]), ]
bestcutoff$Cutoff #0.567

# plot ROC 
plot(roc_logit)  # default
plot(roc_logit, YIndex = T, col = c(2,4)) # Changing color
plot(roc_logit, YIndex = F, col = c(2,4),legend = F) # disable legend
roc_logit$AUC #0.870

#Testing other classifier's performance
# Naive Bayes Classifier
fit.nb.CV <- train(satisfaction ~ Online.boarding + Checkin.service + Departure.Delay.in.Minutes + 
                     Arrival.Delay.in.Minutes + Departure.Arrival.time.convenient + Flight.Distance + Class , 
                   data = satisfaction.train, 
                   method="naive_bayes", 
                   trControl=trainControl(method="cv", number=10))
# Evaluate Performance using Confusion Matrix
actual <- satisfaction.test$satisfaction
# predict class membership
nbPredictClass <- predict(fit.nb.CV, satisfaction.test, type = "raw")
cm <- table(nbPredictClass, actual)
confusionMatrix(nbPredictClass, actual, positive="satisfied")
# Pvalue is less than 0.05 , so these factors are significantly affecting the passenger satisfaction 
# Accuracy is 78.23% which is less than logistic

#Goal 3
#Customer Satisfaction Based on Inflight services

fit <- rpart(satisfaction ~ Age + Class + Inflight.wifi.service + Food.and.drink + Seat.comfort + Inflight.entertainment +  
               Leg.room.service + Cleanliness, # formula, all predictors will be considered in splitting
             data=satisfaction.train, # dataframe used
             method="class",  # treat churn as a categorical variable, default
             control=rpart.control(xval=0, minsplit=1000), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
             parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

fit

rpart.plot(fit, type = 1, extra = 4, main="Classification Passenger Satisfaction")

# extract the vector of predicted class for each observation in satisfaction.train
satisfaction.pred <- predict(fit, satisfaction.train, type="class")
# extract the actual class of each observation in chur.train
satisfaction.actual <- satisfaction.train$satisfaction

confusion.matrix <- table(satisfaction.pred, satisfaction.actual)  
confusion.matrix
addmargins(confusion.matrix)
pt <- prop.table(confusion.matrix)  
pt
#accuracy
pt[1,1] + pt[2,2]

# Testing the model on test data
# extract the predicted class of each observation in satisfaction.test
satisfaction.pred <- predict(fit, satisfaction.test, type="class")
# extract the actual class of each observation in satisfaction.test
satisfaction.actual <- satisfaction.test$satisfaction

# now build the "confusion matrix"
# which is the contingency matrix of predicted vs actual
# use this order: predicted then actual
confusion.matrix <- table(satisfaction.pred, satisfaction.actual)  
confusion.matrix
tp <- confusion.matrix[2,2];tp
tn <- confusion.matrix[1,1];tn
fp <- confusion.matrix[2,1];fp
fn <- confusion.matrix[1,2];fn

accuracy <- (tp + tn)/(tp + tn + fp + fn);accuracy
# TPR = Recall = Sensitivity
TPR <- tp/(fn+tp);TPR
# TNR = Specificity
TNR <- tn/(fp+tn);TNR
# FPR
FPR <- fp/(fp+tn);FPR
# FNR
FNR <- fn/(fn+tp);FNR
