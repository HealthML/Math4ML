## Matrix Norms
[]: # 
[]: # for metric in metrics:
[]: #     clf = FlexibleNearestCentroidClassifier(metric=metric)
[]: #     clf.fit(X_train, y_train)
[]: #     predictions = clf.predict(X_test)
[]: #     print(f"Accuracy with {metric} metric: {accuracy_score(y_test, predictions)}")
[]: # ```
[]: # 
[]: # ---
[]: # 
[]: # ### Conclusion
[]: # 
[]: # The choice of distance metric can significantly affect the performance of the nearest centroid classifier. By experimenting with different metrics, you can gain insights into how they influence classification boundaries and model performance.
[]: # 
[]: # ---
[]: # 
[]: # ### Further Reading
[]: # 
[]: # - [Understanding Distance Metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html)
[]: # - [Nearest Centroid Classifier in Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html)
[]: #
[]: # - [Distance Metrics in Machine Learning](https://towardsdatascience.com/distance-metrics-in-machine-learning-1f3b2a0c4d7e)
[]: # 
[]: # ---
[]: # 
[]: # ### References
[]: # 
[]: # - Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
[]: # - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
[]: # - Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
[]: # 
[]: # ---
[]: # 
[]: # ### License
[]: # 
[]: # This notebook is licensed under the [MIT License](https://opensource.org/licenses/MIT).
[]: # 
[]: # ---
[]: # 
[]: # ### Acknowledgments
[]: # 
[]: # - [Scikit-learn](https://scikit-learn.org/stable/) for providing the machine learning library used in this notebook.
[]: # - [Matplotlib](https://matplotlib.org/) for the visualization tools used in this notebook.