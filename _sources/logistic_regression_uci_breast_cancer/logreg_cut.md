```{code-cell} ipython3
:tags: [hide-input]
a = np.linspace(-8.0,8.0,100) # create 100 points on a line
# Set up the figure
f, ax = plt.subplots(figsize=(7, 6))
linex=np.arange(-9, 9, 0.003)
liney = np.arange(-0.1, 1.1, 0.01)
# xx1, xx2 = np.meshgrid(linex, liney)
# plt.pcolormesh(xx1, xx2, logistic(xx1), cmap='bwr', alpha=0.1)

plt.plot([0,0],[-1,2],':k',alpha=0.8,linewidth=3)
plt.plot(a, logistic(a), 'k', linewidth=5)
# plt.plot(a, 1.0-logistic(a), 'k:', linewidth=5, alpha=0.5)

plt.xlim([-8,8])
plt.ylim([-0.02,1.02])
plt.yticks([0.0,0.5,1.0])
plt.xticks([-8,-4,0,4,8])

# ax.patch.set_facecolor('white')
plt.legend(['decision function',r'$\pi(\mathbf{xw})$','B',"R"])


ax = plt.xlabel(r'$\mathbf{xw}$')
ax = plt.ylabel(r'$p(y=c_1|\mathbf{x})$')

clf = util.LogisticRegression()
clf.fit(X=X.values,y=y.values[:,np.newaxis])
Xw = X.values.dot(clf.w)
bins = np.linspace(-10, 10, 20)
plt.hist(Xw[y.values=='B'], bins, alpha=0.5, label='B', color='b', density=True)
plt.hist(Xw[y.values=='M'], bins, alpha=0.5, label='M', color='r', density=True)
# plt.scatter(Xw, (y.values[:,np.newaxis]=="M") , (y.values[:,np.newaxis]=="M"), size=20)

ax = plt.title("The logistic sigmoid")
# plt.savefig("./uci_breast_cancer/plots/logistic_sigmoid_data.png", dpi=600)
```

```{code-cell} ipython3
:tags: [hide-input]
importlib.reload(util)
f, ax = plt.subplots(figsize=(7,7))
ax, clf = util.plotfun2D_logreg(X,y, threshold=0.5, prob=True, second_line=True)
plt.ylim([8,39.9])
plt.xlim([-0.01,0.45])
# plt.savefig("./uci_breast_cancer/plots/scatter_decision_boundary_secondline.png", dpi=600)
```