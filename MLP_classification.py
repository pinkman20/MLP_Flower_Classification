import numpy as np
import itertools 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix
iris = datasets.load_iris()
#Koristit ćemo samo dvije karakteristike, tako da ih\
#lakše možemo vizualizirati 
X = iris.data[:,[0,2]]
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names 
#get the classes
n_class = len(set(y))
print("Ovaj skup podataka ima ukupno %d klase." %(n_class))
#Grafički prikaz podataka 
plt.figure()#figsize = (10,8))
for i,c,s in (zip(range(n_class),['b','g','r'],['o','^', '*'])):
    ix = y == i 
    #print("ix = " + str(ix))
    #print("y = " + str(y))
    #print("s = " + str(s))
    print("X[:,0][ix] = " + str(X[:,0][ix]))
    print("X[:,1][ix] = " + str(X[:,1][ix]))
    plt.scatter(X[:,0][ix], X[:,1][ix], color = c, marker = s, s = 60, label = target_names[i])
plt.legend(loc = 2, scatterpoints = 1)
plt.xlabel("Feature 1 - " + feature_names[0])
plt.ylabel("Feature 2 - " + feature_names[2])
plt.show()

clf1 = MLPClassifier(hidden_layer_sizes =(10) , activation = 'logistic'  ,max_iter = 100, verbose = "True", random_state =13)
#clf2 = MLPClassifier(hidden_layer_sizes =(10,10) , activation = 'logistic'  ,max_iter = 200, verbose = "True")
#clf3 = MLPClassifier(hidden_layer_sizes =(10,10,10) , activation = 'logistic'  ,max_iter = 300, verbose = "True", random_state =13)
#clf4 = MLPClassifier(hidden_layer_sizes =(20) , activation = 'logistic'  ,max_iter = 400, verbose = "True", random_state =13)
#clf5 = MLPClassifier(hidden_layer_sizes =(20,20) , activation = 'logistic'  ,max_iter = 500, verbose = "True", random_state =13)
#clf6 = MLPClassifier(hidden_layer_sizes =(20,20,20) , activation = 'logistic'  ,max_iter = 600, verbose = "True", random_state =13)
#clf7 = MLPClassifier(hidden_layer_sizes =(30) , activation = 'logistic'  ,max_iter = 700, verbose = "True", random_state =13)
#clf8 = MLPClassifier(hidden_layer_sizes =(10,10) , activation = 'logistic'  ,max_iter = 200, verbose = "True")
#clf9 = MLPClassifier(hidden_layer_sizes =(10,10,10) , activation = 'logistic'  ,max_iter = 300, verbose = "True", random_state =13)
#clf10 = MLPClassifier(hidden_layer_sizes =(20) , activation = 'logistic'  ,max_iter = 400, verbose = "True", random_state =13)
#clf11 = MLPClassifier(hidden_layer_sizes =(20,20) , activation = 'logistic'  ,max_iter = 500, verbose = "True", random_state =13)
#clf12 = MLPClassifier(hidden_layer_sizes =(10,10,10) , activation = 'logistic'  ,max_iter = 300, verbose = "True", random_state =13)
#clf13 = MLPClassifier(hidden_layer_sizes =(20) , activation = 'logistic'  ,max_iter = 400, verbose = "True", random_state =13)
#clf14 = MLPClassifier(hidden_layer_sizes =(20,20) , activation = 'logistic'  ,max_iter = 500, verbose =  "True", random_state =13)
#clf15 = MLPClassifier(hidden_layer_sizes =(20,20,20) , activation = 'logistic'  ,max_iter = 600, verbose = "True", random_state =13)
#clf16 = MLPClassifier(hidden_layer_sizes =(10,10,10) , activation = 'logistic'  ,max_iter = 300, verbose = "True", random_state =13)
#clf17 = MLPClassifier(hidden_layer_sizes =(20) , activation = 'logistic'  ,max_iter = 400, verbose = "True", random_state =13)
#clf18 = MLPClassifier(hidden_layer_sizes =(20,20) , activation = 'logistic'  ,max_iter = 500, verbose = "True")
#clf19 = MLPClassifier(hidden_layer_sizes =(10,10,10) , activation = 'logistic'  ,max_iter = 300, verbose = "True", random_state =13)
#clf20 = MLPClassifier(hidden_layer_sizes =(20) , activation = 'logistic'  ,max_iter = 400, verbose = "True", random_state =13)

#Treniranje/Učenje klasifikatora s podacima za treniranje 
clf1.fit(X,y)
#Predviđanje rezultata na temelju podataka za testiranje 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    plt.figure(figsize=(10, 8)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def plot_decision_boundary(X,y, clf, title = None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                          np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize = (10, 8))
    plt.contourf(xx, yy, Z,alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    
    if title is not None:
        plt.title(title)
    
    plt.xlabel('Feature 1')
    #plt.xlim(3,8)
    plt.ylabel('Feature 2')
    plt.show()
plot_decision_boundary(X,y, clf1)
predicted = clf1.predict(X)
cm = confusion_matrix(y,predicted)
plot_confusion_matrix(cm, classes=iris.target_names,\
                      title='Confusion matrix, without normalization')

