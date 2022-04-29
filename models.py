from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def random_forest(x_train, y_train):
  model = RandomForestClassifier() 
  model.fit(x_train, y_train)
  return model

def log_reg(x_train, y_train):
  model = LogisticRegression()
  model.fit(x_train, y_train)
  return model

def tree(x_train, y_train):
  model = DecisionTreeClassifier()
  model.fit(x_train, y_train)
  return model

