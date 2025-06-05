import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn 
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import dagshub

dagshub.init(repo_owner='vinayak910', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_url("https://dagshub.com/vinayak910/mlflow-dagshub-demo.mlflow")

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# Step 3: Split the dataset into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 10

#####giving name to the experiment 

# 1st method -> 
mlflow.set_experiment("iris-dec-tree")  


#2nd method-> pass experiment id of the experiment from mlflow ui into start_run
#  experiment_id=916318673609904012


####### apply mlflow 

with mlflow.start_run():
    # now inside this, everything will be logged 
    rf = DecisionTreeClassifier(max_depth= max_depth)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test , y_pred)

    cm = confusion_matrix(y_test , y_pred)


    # logging params and accuracy

    mlflow.log_metric("accuracy" , accuracy)
    mlflow.log_param("max_depth" , max_depth)
    mlflow.log_param("confusion_matrix" , cm)
    print("accuracy", accuracy)
    
    plt.figure(figsize = (6, 6))
    sns.heatmap(cm , annot = True , fmt = 'd' , cmap = 'Blues' , xticklabels= iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("confusion_matrix.png")

    # saving the plot and then logging 
    mlflow.log_artifact('confusion_matrix.png')

    # logging the code 
    mlflow.log_artifact(__file__)


    # logging model 
    mlflow.sklearn.log_model(rf , "decision tree")

    # logging tags 
    mlflow.set_tag('author' , 'Vinayak')
    mlflow.set_tag('model' , 'decision tree')