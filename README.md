# Random Forest Classification for Car Safety Prediction

This project demonstrates the use of the Random Forest algorithm to predict the safety of cars based on various features. The analysis is conducted using the Car Evaluation Data Set from the UCI Machine Learning Repository.


## Introduction
Random Forest is a supervised machine learning algorithm based on ensemble learning. This project builds two Random Forest Classifier models to predict car safety:
- Model with 10 decision trees.
- Model with 100 decision trees.

Additionally, the feature selection process identifies important features, rebuilds the model, and examines its effect on accuracy.

## Algorithm Intuition
Random Forest combines multiple decision trees to enhance prediction accuracy:
1. **Training Stage**: Build a forest of decision trees using random subsets of features and data.
2. **Prediction Stage**: Aggregate predictions from all trees via majority voting.

## Advantages and Disadvantages
### Advantages
- Versatile for classification and regression.
- Reduces overfitting through averaging predictions.
- Handles missing values and provides feature importance scores.

### Disadvantages
- Computationally expensive with a large number of trees.
- Difficult to interpret compared to individual decision trees.

## Feature Selection
Random Forest ranks feature importance by evaluating the impact on out-of-bag (OOB) error. Less important features are removed to improve model efficiency.

## Problem Statement
The goal is to predict car safety using features like buying cost, maintenance cost, and safety ratings.

## Dataset Description
The dataset contains 1,728 instances and 7 attributes:
- **buying**: Buying cost.
- **maint**: Maintenance cost.
- **doors**: Number of doors.
- **persons**: Capacity in terms of persons.
- **lug_boot**: Size of luggage boot.
- **safety**: Safety rating.
- **class**: Target variable (safety category).

Dataset Source: [Car Evaluation Data Set](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/random-forest-car-safety.git
   cd random-forest-car-safety
