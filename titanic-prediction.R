
#install.packages("neuralnet")

library(neuralnet)
library(ggplot2)
library(dplyr)
 



titanic_data <- read.csv("Titanic-Dataset.csv")

# Convert female - male to '0' - '1'
titanic_data$Sex <- as.numeric(factor(titanic_data$Sex, levels = c("female", "male")))

# Handle missing values in 'Age' (e.g., by filling with the mean)
titanic_data$Age[is.na(titanic_data$Age)] <- mean(titanic_data$Age, na.rm = TRUE)


# summary of dataset
summary(titanic_data)
str(titanic_data)

# Visualize the distribution of the target variable 'Survived'
ggplot(titanic_data, aes(x = factor(Survived))) +
  geom_bar() +
  labs(title = "Distribution of Survived",
       x = "Survived",
       y = "Frequency")

# Visualization
ggplot(titanic_data, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  labs(title = "Survived by Pclass",
       x = "Pclass",
       y = "Frequency",
       fill = "Survived")

ggplot(titanic_data, aes(x = factor(Sex), fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  labs(title = "Survived by Sex",
       x = "Sex",
       y = "Frequency",
       fill = "Survived")

ggplot(titanic_data, aes(x = factor(SibSp), fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  labs(title = "Survived by SibSp",
       x = "SibSp",
       y = "Frequency",
       fill = "Survived")

ggplot(titanic_data, aes(x = factor(Parch), fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  labs(title = "Survived by Parch",
       x = "Parch",
       y = "Frequency",
       fill = "Survived")


set.seed(123)

# train test split
sample_index <- sample(1:nrow(titanic_data), 0.7*nrow(titanic_data))
train_data <- titanic_data[sample_index,]
test_data <- titanic_data[-sample_index,]



# NN Model - logistic regression
nnlr_model <- neuralnet(Survived ~ Pclass + Sex + Age+ SibSp, data = train_data, hidden = c(0), linear.output = F)


print(nnlr_model)

# predictions
predictions <- predict(nnlr_model, test_data)

# predictions to binary (0 or 1)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)



# Accuracy
accuracyLR <- sum(predicted_classes == test_data$Survived) / nrow(test_data)
cat("Accuracy:", accuracyLR)

View(nnlr_model)


plot(nnlr_model, rep = "best")

# Define a function to calculate the classification error
classification_error <- function(predictions, actual) {
  mean(ifelse(predictions > 0.5, 1, 0) != actual)
}


# Define a vector to store the error values
error_values <- numeric(0)

# Train the model with recording of error values
for (i in 1:100) {
  nnlr_model <- neuralnet(Survived ~ Pclass + Sex + Age + SibSp, 
                          data = train_data, hidden = c(0), linear.output = FALSE)
  predictions <- as.vector(predict(nnlr_model, train_data))
  error <- classification_error(predictions, train_data$Survived)
  error_values <- c(error_values, error)
}

# Plot the error vs iteration
plot(1:100, error_values, type = "l", 
     main = "Error vs Iteration", xlab = "Iteration", ylab = "Error")


#-------------------------------------------------------------------------------------------------------------------------------------------


# NN Model -  MLP
nnMLP_model <- neuralnet(Survived ~ Pclass + Sex + Age+ SibSp, data = train_data, hidden = c(4), linear.output = F)


print(nnMLP_model)

# predictions
predictions <- predict(nnMLP_model, test_data)

# predictions to binary (0 or 1)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Accuracy
accuracyMLP <- sum(predicted_classes == test_data$Survived) / nrow(test_data)
cat("Accuracy:", accuracyMLP)

plot(nnMLP_model)
View(nnMLP_model)

# Define a function to calculate the classification error
classification_error <- function(predictions, actual) {
  mean(ifelse(predictions > 0.5, 1, 0) != actual)
}


# Define a vector to store the error values
error_values <- numeric(0)

# Train the model with recording of error values
for (i in 1:100) {
  nnlr_model <- neuralnet(Survived ~ Pclass + Sex + Age + SibSp, 
                          data = train_data, hidden = c(0), linear.output = FALSE)
  predictions <- as.vector(predict(nnlr_model, train_data))
  error <- classification_error(predictions, train_data$Survived)
  error_values <- c(error_values, error)
}

# Plot the error vs iteration
plot(1:100, error_values, type = "l", 
     main = "Error vs Iteration", xlab = "Iteration", ylab = "Error")




#--------------------------------------------------------------------------------------------

# NN Model -  MLP
nnMLP2_model <- neuralnet(Survived ~ Pclass + Sex + Age+ SibSp, data = train_data, hidden = c(2,2), linear.output = F)


print(nnMLP2_model)
summary(nnMLP2_model)

# predictions
predictions <- predict(nnMLP2_model, test_data)

# predictions to binary (0 or 1)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Accuracy
accuracyMLP2 <- sum(predicted_classes == test_data$Survived) / nrow(test_data)
cat("Accuracy:", accuracyMLP2)

plot(nnMLP2_model)
View(nnMLP2_model)

# Define a function to calculate the classification error
classification_error <- function(predictions, actual) {
  mean(ifelse(predictions > 0.5, 1, 0) != actual)
}


# Define a vector to store the error values
error_values <- numeric(0)

# Train the model with recording of error values
for (i in 1:100) {
  nnlr_model <- neuralnet(Survived ~ Pclass + Sex + Age + SibSp, 
                          data = train_data, hidden = c(0), linear.output = FALSE)
  predictions <- as.vector(predict(nnlr_model, train_data))
  error <- classification_error(predictions, train_data$Survived)
  error_values <- c(error_values, error)
}

# Plot the error vs iteration
plot(1:100, error_values, type = "l", 
     main = "Error vs Iteration", xlab = "Iteration", ylab = "Error")


#---------------------------------------------------------------------------------------------------------





# barplot comparisions


categories <- c("LR", "1 hidden layer 4 nodes", "2 hidden layers 2 nodes each")
values <- c(accuracyLR, accuracyMLP, accuracyMLP2)


barplot(values, names.arg = categories, col = "pink", main = "Bar Plot", xlab = "Neural Network", ylab = "Accuracy", ylim = c(0, 1))

     