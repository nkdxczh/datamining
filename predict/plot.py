import sys
sys.path.insert(0, '/home/jason/datamining/project')

from feature.select import *
from feature.read import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

from matplotlib import pyplot

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    print('testing gridsearch...')  
    data_file = "/home/jason/datamining/data/train_combine.csv"
    
    print('reading training and testing data...')    
    X, Y = read_data(data_file) 
    nX = normalize(X)
    sX = scale(X)

    print('selecting features...')   
    select_model = feature_select_et(nX, Y)
    nXef = select_model.transform(nX)
    print(len(nXef[0]))

    select_model = feature_select_rf(nX, Y)
    nXrf = select_model.transform(nX)
    print(len(nXrf[0]))

    select_model = feature_select_ls(nX, Y)
    nXls = select_model.transform(nX)
    print(len(nXls[0]))

    select_model = feature_select_et(sX, Y)
    sXef = select_model.transform(nX)
    print(len(sXef[0]))

    select_model = feature_select_rf(sX, Y)
    sXrf = select_model.transform(nX)
    print(len(sXrf[0]))

    select_model = feature_select_ls(sX, Y)
    sXls = select_model.transform(nX)
    print(len(sXls[0]))

    print('pca features...')   
    
    pca = feature_pca(nXef)
    nXef = pca.transform(nXef)
    nnXef = normalize(nXef)
    nsXef = scale(nXef)
    
    pca = feature_pca(nXrf)
    nXrf = pca.transform(nXrf)
    nnXrf = normalize(nXrf)
    nsXrf = scale(nXrf)
    
    '''pca = feature_pca(nXls)
    nXls = pca.transform(nXls)
    nnXls = normalize(nXls)
    nsXls = scale(nXls)'''
    
    pca = feature_pca(sXef)
    sXef = pca.transform(sXef)
    snXef = normalize(sXef)
    ssXef = scale(sXef)
    
    pca = feature_pca(sXrf)
    sXrf = pca.transform(sXrf)
    snXrf = normalize(sXrf)
    ssXrf = scale(sXrf)
    
    pca = feature_pca(sXls)
    sXls = pca.transform(sXls)
    snXls = normalize(sXls)
    ssXls = scale(sXls)

    '''X1 = []
    X2 = []
    for i in range(len(Y)):
        if Y[i] == 1:
            X1.append(X[i,:])
        else:
            X2.append(X[i,:])'''

    print('draw features...')  

    fig = plt.figure()
   
    px = nnXef[:,0]
    py = nnXef[:,1]
    pz = nnXef[:,2]
    ax = fig.add_subplot(431, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    px = nnXrf[:,0]
    py = nnXrf[:,1]
    pz = nnXrf[:,2]
    ax = fig.add_subplot(432, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    '''px = nnXls[:,0]
    py = nnXls[:,1]
    pz = nnXls[:,2]
    ax = fig.add_subplot(433, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')'''
   
    px = nsXef[:,0]
    py = nsXef[:,1]
    pz = nsXef[:,2]
    ax = fig.add_subplot(434, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    px = nsXrf[:,0]
    py = nsXrf[:,1]
    pz = nsXrf[:,2]
    ax = fig.add_subplot(435, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    '''px = nsXls[:,0]
    py = nsXls[:,1]
    pz = nsXls[:,2]
    ax = fig.add_subplot(436, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')'''
   
    px = ssXef[:,0]
    py = ssXef[:,1]
    pz = ssXef[:,2]
    ax = fig.add_subplot(437, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    px = ssXrf[:,0]
    py = ssXrf[:,1]
    pz = ssXrf[:,2]
    ax = fig.add_subplot(438, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    px = ssXls[:,0]
    py = ssXls[:,1]
    pz = ssXls[:,2]
    ax = fig.add_subplot(439, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    px = snXef[:,0]
    py = snXef[:,1]
    pz = snXef[:,2]
    ax = fig.add_subplot(4,3,10, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    px = snXrf[:,0]
    py = snXrf[:,1]
    pz = snXrf[:,2]
    ax = fig.add_subplot(4,3,11, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
   
    px = snXls[:,0]
    py = snXls[:,1]
    pz = snXls[:,2]
    ax = fig.add_subplot(4,3,12, projection='3d')
    ax.scatter(px, py, pz, c=Y, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
