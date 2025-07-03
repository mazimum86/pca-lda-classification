# Linear Discriminant Analysis (LDA) with SVM on the Wine Dataset

# Load dataset
dataset <- read.csv('../data/wine.csv')

# Split dataset into training and test sets
library(caTools)
set.seed(123)
split <- sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
dataset_train <- subset(dataset, split == TRUE)
dataset_test  <- subset(dataset, split == FALSE)

# Feature scaling (exclude the label column)
dataset_train[-14] <- scale(dataset_train[-14])
dataset_test[-14]  <- scale(dataset_test[-14])

# Apply LDA for dimensionality reduction
library(MASS)
lda_model <- lda(Customer_Segment ~ ., data = dataset_train)

# Transform datasets using LDA projection
dataset_train <- as.data.frame(predict(lda_model, dataset_train))
dataset_test  <- as.data.frame(predict(lda_model, dataset_test))

# Keep only LD1, LD2, and class label
dataset_train <- dataset_train[c("x.LD1", "x.LD2", "class")]
dataset_test  <- dataset_test[c("x.LD1", "x.LD2", "class")]

# Train SVM classifier
library(e1071)
classifier <- svm(
  formula = class ~ .,
  data = dataset_train,
  type = 'C-classification',
  kernel = 'linear'
)

# Predict on the test set
y_pred <- predict(classifier, newdata = dataset_test[-3])

# Confusion matrix
cm <- table(Actual = dataset_test$class, Predicted = y_pred)
print(cm)

# Function to visualize decision boundary
visualize_results <- function(set, title) {
  X1 <- seq(min(set[,1]) - 1, max(set[,1]) + 1, by = 0.01)
  X2 <- seq(min(set[,2]) - 1, max(set[,2]) + 1, by = 0.01)
  X_grid <- expand.grid(X1, X2)
  colnames(X_grid) <- c("x.LD1", "x.LD2")
  y_grid <- as.numeric(predict(classifier, newdata = X_grid))
  
  plot(set[-3],
       main = title,
       xlab = 'LD1',
       ylab = 'LD2',
       xlim = range(X1),
       ylim = range(X2))
  
  contour(X1, X2, matrix(y_grid, length(X1), length(X2)), add = TRUE)
  points(X_grid, pch = '.', col = ifelse(y_grid == 1, 'tomato',
                                         ifelse(y_grid == 2, 'springgreen', 'skyblue')))
  points(set[-3], pch = 21, col = 'black',
         bg = ifelse(set[,3] == 1, 'darkred',
                     ifelse(set[,3] == 2, 'darkgreen', 'darkblue')))
}

# Visualize results on the training set
visualize_results(dataset_train, 'LDA - Training Set')

# Visualize results on the test set
visualize_results(dataset_test, 'LDA - Test Set')
