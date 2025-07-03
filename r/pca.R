# PCA with SVM Classification on Wine Dataset

# Load the dataset
dataset <- read.csv('../data/wine.csv')

# Split the dataset into training and test sets
library(caTools)
set.seed(123)
split <- sample.split(dataset, SplitRatio = 0.8)
dataset_train <- subset(dataset, split == TRUE)
dataset_test  <- subset(dataset, split == FALSE)

# Feature scaling (excluding the target variable at column 14)
dataset_train[, -14] <- scale(dataset_train[, -14])
dataset_test[, -14]  <- scale(dataset_test[, -14])

# Apply PCA to reduce to 2 components
library(caret)
pca <- preProcess(dataset_train[, -14], method = 'pca', pcaComp = 2)
dataset_train <- predict(pca, dataset_train)
dataset_test  <- predict(pca, dataset_test)

# Rearranging columns: PC1, PC2, then target
dataset_train <- dataset_train[, c("PC1", "PC2", "Customer_Segment")]
dataset_test  <- dataset_test[, c("PC1", "PC2", "Customer_Segment")]

# Fit SVM classifier with a linear kernel
library(e1071)
classifier <- svm(formula = Customer_Segment ~ .,
                  data = dataset_train,
                  type = 'C-classification',
                  kernel = 'linear')

# Predict the test set
y_pred <- predict(classifier, newdata = dataset_test)

# Generate confusion matrix
cm <- table(dataset_test$Customer_Segment, y_pred)
print("Confusion Matrix:")
print(cm)

# Function to visualize results
visualize_results <- function(data, title) {
  X1 <- seq(min(data[, 1]) - 1, max(data[, 1]) + 1, by = 0.01)
  X2 <- seq(min(data[, 2]) - 1, max(data[, 2]) + 1, by = 0.01)
  grid <- expand.grid(PC1 = X1, PC2 = X2)
  y_grid <- predict(classifier, newdata = grid)
  
  plot(data[, -3],
       main = title,
       xlab = 'PC1',
       ylab = 'PC2',
       xlim = range(X1),
       ylim = range(X2))
  
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  points(grid,
         pch = '.',
         col = ifelse(y_grid == 1, 'tomato',
                      ifelse(y_grid == 2, 'springgreen', 'skyblue')))
  points(data[, -3],
         pch = 21,
         bg = ifelse(data[, 3] == 1, 'red',
                     ifelse(data[, 3] == 2, 'darkgreen', 'blue')))
}

# Visualize training set
visualize_results(dataset_train, "PCA - Training Set")

# Visualize test set
visualize_results(dataset_test, "PCA - Test Set")
