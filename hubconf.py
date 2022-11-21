import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_blobs,make_circles,load_digits
from sklearn.cluster import KMeans
import numpy as np

def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  return X,y

def get_data_circles(n_points=100):
  X, y = make_circles(n_samples = n_points)
  return X,y

def get_data_mnist():
  X,y = load_digits(return_X_y=True)
  #print(X.shape)
  return X,y

def build_kmeans(X=None,k=10):
  kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
  return kmeans

def assign_kmeans(km=None,X=None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values
  return h,c,v
