import numpy as np               # efficient matrix-vector operations
import numpy.linalg as la        # linear algebra (solvers etc.)
import pandas as pd              # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns            # data visualization  
import matplotlib.pyplot as plt  # basic plotting
# from scipy.stats import norm  # not used
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
import logistic_regression
import time
sns.set(style="whitegrid")        # set the figure default style

sns.set(font_scale=1.5)          # bigger fonts in images

def plot_logistic():
    a = np.linspace(-8.0,8.0,100) # create 100 points on a line
    # Set up the figure
    f, ax = plt.subplots(figsize=(7, 6))
    linex=np.arange(-9, 9, 0.003)
    liney = np.arange(-0.1, 1.1, 0.01)
    xx1, xx2 = np.meshgrid(linex, liney)
    plt.pcolormesh(xx1, xx2, logistic_regression.logistic(xx1), cmap='bwr', alpha=0.1)

    plt.plot([0,0],[-1,2],':k',alpha=0.8,linewidth=3)
    plt.plot(a, logistic_regression.logistic(a), 'k', linewidth=5)
    # plt.plot(a, 1.0-logistic(a), 'k:', linewidth=5, alpha=0.5)

    plt.xlim([-8,8])
    plt.ylim([-0.02,1.02])
    plt.yticks([0.0,0.5,1.0])
    plt.xticks([-8,-4,0,4,8])

    ax.patch.set_facecolor('white')
    plt.legend(['decision function','$\pi(\mathbf{xw})$'])

    ax = plt.xlabel('$\mathbf{xw}$')
    ax = plt.ylabel('$p(y=c_1|\mathbf{x})$')


    bins = np.linspace(-10, 10, 20)

    # plt.scatter(Xw, (y.values[:,np.newaxis]=="M") , (y.values[:,np.newaxis]=="M"), size=20)

    ax = plt.title("The logistic sigmoid")
    plt.savefig("./uci_breast_cancer/plots/logistic_sigmoid.png", dpi=600)
    
def load_data(testing_data=False, columns=None):
    data = pd.read_csv('./../datasets/breast_cancer_data/data_processed.csv') #'./uci_breast_cancer/input/data.csv')
    # y has the labels and x holds our features
    y = data["diagnosis"]      # M or B 
    x = data.drop(['diagnosis'],axis = 1 )
    if columns:
        x = x[columns]
    x['bias'] = np.ones(x.shape[0])
    # split data train 70 % and test 30 %
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    if testing_data:
        return x_test, y_test
    else:
        return x_train, y_train


def scatter_plot_kde2(X,y):
    ax = plt.gca()
    
    plt.scatter(X.concavity_mean[y=="M"], X.texture_mean[y=="M"], alpha=0.8, color="r")
    plt.scatter(X.concavity_mean[y=="B"], X.texture_mean[y=="B"], alpha=0.8, color="b")
    plt.legend(["$c_1$ (M)", "$c_2$ (B)"])
    # Draw the two density plots
    #ax = sns.kdeplot(X.concavity_mean[y=="M"], X.texture_mean[y=="M"],cmap="Reds", shade=True, shade_lowest=False, alpha=0.4)
    #ax = sns.kdeplot(X.concavity_mean[y=="B"], X.texture_mean[y=="B"], cmap="Blues", shade=True, shade_lowest=False,alpha=0.4)
    plt.xlabel("$x_1$    (" + X.columns[0]+")")
    plt.ylabel("$x_2$    (" + X.columns[1]+")")
    return ax


def eval_optimizer(X, y, steep=False, irls=False, grad=True):
    ax = plt.gca()
    max_feature = 3
    
    clf = logistic_regression.LogisticRegression(learning_rate = 0.001, max_iter=10000)
    
    # fit the model using IRLS to get bias term correct and a good w
    clf.optimizer = "IRLS"
    clf.verbose = False
    clf.debug = True
    clf, res = clf.fit(X.values[:,0:max_feature],y.values[:,np.newaxis])
    w_init = clf.w.copy()
    w_opt = w_init.copy()
    # print (w_init[:,0])
    
    h=0.5
    minx1 = -10
    maxx1 = 50
    minx2 = -0.5
    maxx2 = 1.5

    xx1, xx2 = np.meshgrid(np.arange(minx1, maxx1, 3.0), np.arange(minx2, maxx2, 0.03))
    w_grid = np.c_[xx1.ravel(), xx2.ravel(), w_init[-1,0] * np.ones(xx2.ravel().shape[0])]
    Z = np.zeros(w_grid.shape[0])
    
    for i in np.arange(w_grid.shape[0]):
        clf.w = w_grid[i,0:max_feature,np.newaxis]
        Z[i] = clf.objective(X=X.values[:,0:max_feature],y=y.values[:,np.newaxis])
            
    Z = Z.reshape(xx1.shape)
    
    plt.pcolormesh(xx1, xx2, Z, cmap="viridis")
    
    w_irls = np.concatenate(res['w'],1)
    
    plt.colorbar()
    plt.axhline(y=0, xmin=minx1, xmax=maxx1, color='w')
    plt.axvline(x=0, ymin=minx2, ymax=maxx2, color='w')

    
    plt.plot(w_opt[0,0],w_opt[1,0],'mo', markersize=15)
    # fit the model using steepest descent
    
    if steep:
        # fit the model using steepest descent
        clf.optimizer = "steep"
        clf.learning_rate=0.0002
        clf.max_iter = 200000
        w_init[:] = 0.0
        clf, opt = clf.fit(X.values[:,0:max_feature],y.values[:,np.newaxis], w_init=None)
        w_steep = np.concatenate(opt['w'],1)
        
        plt.plot(w_steep[0][::1001],w_steep[1][::1001],'y.-', markersize=5, linewidth=3)
    if irls:
        plt.plot(w_irls[0],w_irls[1], '.-', markersize=15, linewidth=3, color="r")
    plt.xlabel("$w_1$    (" + X.columns[0]+")")
    plt.ylabel("$w_2$    (" + X.columns[1]+")")
    return ax


def eval_optimizer1D(X, y, taylor1=False, taylor2=False):
    ax = plt.gca()
    max_feature = 3
    
    clf = logistic_regression.LogisticRegression(learning_rate = 0.001, max_iter=10000)
    
    # fit the model using IRLS to get bias term correct and a good w
    clf.optimizer = "IRLS"
    clf.verbose = False
    clf.debug = True
    clf, res = clf.fit(X.values[:,0:max_feature],y.values[:,np.newaxis])
    w_init = clf.w.copy()
    w_opt = w_init.copy()
    # print (w_init[:,0])
    
    minx = 27
    maxx = 40

    xx = np.arange(minx, maxx, 0.3)
    plt.xlim([minx,maxx])
    
    Z = np.zeros(xx.shape[0])
    dZ = np.zeros(xx.shape[0])
    ddZ = np.zeros(xx.shape[0])

    def eval1D(w1):
        ww = clf.w.copy()
        clf.w[0]=w1
        ret = clf.objective(X=X.values[:,0:max_feature],y=y.values[:,np.newaxis])
        clf.w = ww
        return ret
    
    def derivative1D(w1):
        ww = clf.w.copy()
        clf.w[0]=w1
        ret = clf.gradient(X=X.values[:,0:max_feature],y=y.values[:,np.newaxis])
        clf.w = ww
        return ret[0,0]     

    def hessian1D(w1):
        ww = clf.w.copy()
        clf.w[0]=w1
        ret = clf.hessian(X=X.values[:,0:max_feature],y=y.values[:,np.newaxis])
        clf.w = ww
        return ret[0,0]     

    
    
    for i in np.arange(xx.shape[0]):
        Z[i] = eval1D(xx[i])
        dZ[i] = derivative1D(xx[i])
        ddZ[i] = hessian1D(xx[i])
          
    Z = Z.reshape(xx.shape)
    
    plt.ylim([Z.min()-3,Z.max()])
    
    
    # plt.plot(xx, Z+xx*dZ)
    # plt.plot(xx, ddZ)


    plt.xlabel("$w_1$    " + X.columns[0])
    plt.ylabel("$L$")
    
    def taylor1_1D(x,a):
        fa = eval1D(a)
        da = derivative1D(a)
        return fa+(x-a)*da
    
    def taylor2_1D(x,a):
        dda = hessian1D(a)
        return taylor1_1D(x,a) + 0.5 * (x-a)*(x-a) * dda
    
    def solve_taylor2_1D(x):
        dx = derivative1D(x)
        ddx = hessian1D(x)
        return x - dx/ddx
        
    
    a = 28 # where we are 
    
    plt.plot(a,eval1D(a),'.k', markersize=25) # where we are
    plt.plot(w_opt[0],eval1D(w_opt[0]),'om', markersize=15) # the minimum
    plt.plot(xx, Z, 'm', linewidth=3)
    
    legend = [ '$w_1^{t}$','$w_1^{opt}$', '$L(w_1)$']
    if taylor1:
        plt.plot(xx,taylor1_1D(xx,a),':k', linewidth=4) 
        legend.append("$L(w_1^{t}) + (w_1 - w_1^{t})\cdot \partial/\partial w_1 L(w_1^{t})$")
    if taylor2:
        plt.plot(xx,taylor2_1D(xx,a),':r', linewidth=4) 
        solution_a = solve_taylor2_1D(a) 
        plt.plot(solution_a,taylor2_1D(solution_a,a),'r.', markersize=25) # the minimum of the taylor exapnsion
        legend.append("$L(w_1^{t}) + (w_1 - w_1^{t})\cdot \partial/\partial w_1L(w_1^{t})   + 0.5*(w_1 - w_1^{t})^2\cdot \partial^2/\partial^2 w_1L(w_1^{t})$")
        legend.append("$w_1^{t+1}$")
    # plt.legend(legend)
    return ax




def plotfun2D_logreg(X,y,X_test=None,y_test=None, h = 0.003, threshold=0.5, cmap='bwr', prob=True, second_line=False):
    x=X
    minx1 = x.values[:,0].min()-0.05
    maxx1 = x.values[:,0].max()+0.05
    minx2 = x.values[:,1].min()-5
    maxx2 = x.values[:,1].max()+5
    linex=np.arange(minx1, maxx1, h*(maxx1-minx1))
    xx1, xx2 = np.meshgrid(linex, np.arange(minx2, maxx2, h*(maxx2-minx2)))
    X_grid = np.c_[xx1.ravel(), xx2.ravel(), np.ones(xx2.ravel().shape[0])]
    
    y_vals = np.unique(y)
    y = y==y_vals[1]
    clf = logistic_regression.LogisticRegression()
    clf.fit(X=X.values,y=y.values[:,np.newaxis])
    w = clf.w.copy()
    
    def normal_line(x, threshold=0.5, w=w):
        """
        w[2,0]+x*w[0,0]+y*w[1,0] = logit(threshold)
        where logit = logistic^{-1}
        returns y = logit(threshold)/w[1,0]-(w[2,0]+x*w[0,0])/w[1,0]
        """
        return (logistic_regression.logit(threshold)-w[2,0]-x*w[0,0])/w[1,0]
    
    Z = clf.predict_proba(X_grid)
    Z = Z.reshape(xx1.shape)
    
    ax = plt.gca()
    if prob:
        plt.pcolormesh(xx1, xx2, Z, cmap=cmap, alpha=0.05)
    else:
        plt.pcolormesh(xx1, xx2, 1.0*(Z>=threshold), cmap=cmap, alpha=0.1)
    
    # Plot also the training points
    plt.scatter(x.concavity_mean[y], x.texture_mean[y], alpha=0.8, color="r")
    plt.scatter(x.concavity_mean[~y], x.texture_mean[~y], alpha=0.8, color="b")
    legend = ["$c_1$ (M)", "$c_2$ (B)"]
    if X_test is not None:
        plt.scatter(X_test.concavity_mean, X_test.texture_mean, alpha=1, color="w",marker='o',edgecolors='k', s=100)
        legend = ["$c_1$ (M)", "$c_2$ (B)", "?"]

    plt.legend(legend)
    plt.plot(linex,normal_line(linex, threshold=threshold),':k', linewidth=3, alpha=0.8)
    #plt.scatter(x.values[:, 0], x.values[:, 1], c=(y), edgecolors='k', alpha=0.8, cmap='bwr')
    # plt.scatter(x[y].values[:, 0], x[y].values[:, 1], c='r',cmap='bwr', alpha=0.8)
    # plt.scatter(x[~y].values[:, 0], x[~y].values[:, 1], c='b',cmap='bwr', alpha=0.8)
    if second_line:
        w_alt = w.copy()
        w_alt[0]-=0.1
        w_alt[1]-=0.15
        w_alt[2]+=3
        plt.plot(linex,normal_line(linex, threshold=threshold, w=w_alt),':k', linewidth=3, alpha=0.8)
    
    plt.xlabel("$x_1$    (" + x.columns[0]+")")
    plt.ylabel("$x_2$    (" + x.columns[1]+")")


    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    ax.grid('off')
    ax.patch.set_facecolor('white')
    
    return ax, clf
    

def plot_confusion_matrix(x, y, x_test, y_test, threshold=0.5):
    print("number samples: " + str(x_test.shape[0]))
    print("number M: " + str((y_test=="M").sum()))
    print("number B: " + str((y_test=="B").sum()))


    clf = logistic_regression.LogisticRegression()
    clf.fit(X=x.values,y=y.values[:,np.newaxis])
    predictions=clf.predict(x_test.values, threshold=threshold)
    ac = accuracy_score(y_test,predictions)
    cm = pd.DataFrame(confusion_matrix(y_test,predictions, labels=["B", "M"]),index=["B", "M"],columns=["B", "M"])
    sns.heatmap(cm,annot=True,fmt="d")
    plt.xlabel("predicted diagnosis")
    plt.ylabel("true diagnosis")
    print('Accuracy   : {:.3f}'.format(ac))
    print("Sensitivity: {:.3f}".format( cm["M"]["M"] / (cm["M"]["M"]+cm["B"]["M"])))
    print("Specificity: {:.3f}".format( cm["M"]["M"] / (cm["M"]["M"]+cm["M"]["B"])))