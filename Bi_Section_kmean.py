from kneed import KneeLocator
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score
# from sklearn.datasets import make_blobs
import pandas as pd
from tkinter import messagebox


def Bisection_Mean_function(df):
    sse_list = []
    for k in range(1, 11):
        bisection = BisectingKMeans(n_clusters=k)
        bisection.fit(df)  # Fit the model to the scaled data
        sse_list.append(bisection.inertia_)

    # Find the elbow point
    k1 = KneeLocator(range(1, 11), sse_list, curve='convex', direction='decreasing')
    optimal_num_clusters = k1.elbow
    

    # print("Optimal number of clusters:", optimal_num_clusters)
    best_kmean =BisectingKMeans(n_clusters=optimal_num_clusters, init='random', random_state=1,max_iter=300,n_init=10)
    best_kmean.fit_predict(df) 


    #accurcy 
    cluster_labels = best_kmean.labels_
    silhouette_avg = silhouette_score(df,cluster_labels)
    messagebox.showinfo("Info",silhouette_avg)
    
    # print(silhouette_avg)

    



