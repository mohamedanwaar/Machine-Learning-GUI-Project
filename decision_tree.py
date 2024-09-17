import  pandas  as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , mean_squared_error ,recall_score,precision_score,confusion_matrix
from tkinter import messagebox


import tkinter as tk
from tkinter import ttk

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def decision_tree_funchon(df,targetCoulmn):
        

        x = df.drop(targetCoulmn, axis=1)  # Adjust 'target_column' to your target variable
        y = df[targetCoulmn]  # Adjust 'target_column' to your target variable

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

        # Instantiate and fit the model
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
         
        accuracy = accuracy_score(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision =precision_score(y_test, y_pred)


        messagebox.showinfo("Info", 
                                "Accuracy: " + str(accuracy) + "\n" +
                                "MES: " + str(MSE) + "\n" +
                                "Confusion Matrix: " + "\n" + str(confusion) + "\n" +
                                "Recall Score: " + str(recall) + "\n" +
                                "Precision Score: " + str(precision))
        





           # Plot the decision tree
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10),dpi=300)
        plot_tree(model, filled=True, feature_names=list(x.columns), class_names=[str(c) for c in model.classes_], ax=ax)
        fig.tight_layout()
        
        plt.close(fig)  # Close the figure to prevent it from displaying immediately

        fig.savefig("imagename2.png") 
       
       
 
        # fig, ax = plt.subplots(figsize=(12, 12))
        # plot_tree(model, filled=True, feature_names=x.columns, class_names=[str(c) for c in model.classes_], ax=ax)
        # fig.tight_layout()
        # plt.close(fig)  # Close the figure to prevent it from displaying immediately
        # plt.savefig('decision_tree.png')  # Save the plot as a PNG image
        
                        

        