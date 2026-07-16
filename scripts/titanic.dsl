# SimpleML DSL script for Titanic classification pipeline
load titanic from "titanic.csv"
preprocess titanic using "Survived"
train titanic_model using logistic_regression on titanic
evaluate titanic_model using metrics
predict titanic_model using {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}
summary titanic_model
