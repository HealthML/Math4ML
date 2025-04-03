# Mathematics for Machine Learning

Machine learning (ML) and data science involve algorithms that automatically learn patterns and relationships directly from data. At their core, many ML methods rely heavily on fundamental mathematical concepts—especially linear algebra, calculus, probability, and optimization—to understand, analyze, and make predictions from data.

Our assumption is that the reader is already familiar with the basic concepts of multivariable calculus, linear algebra and probability theory (at the level of Mathematik 2 and Mathematik 3 in the ITSE Bachelor).

This work is based on Mathematics for Machine Learning ([math4ml](https://github.com/gwthomas/math4ml)) by Garrett Thomas, Department of Electrical Engineering and Computer Sciences, University of California, Berkeley, downloaded on February 23, 2025. The original note by Garrett Thomas, which includes the permissive phrase "You are free to distribute this document as you wish.", has been used as a frame that provides the mathematical topics. However, it has been significantly expanded to include more explanations, more detailed proofs, and applications from the machine learning domain. This expanded version, including additional explanations, proofs, and applications, is distributed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).



## Why Mathematical Foundations?

This book carefully introduces and builds upon essential mathematical foundations necessary for modern machine learning:

- **Linear Algebra** (Chapters on vector spaces, linear maps, norms, inner products, eigen-decompositions, and matrix factorizations) provides the language and tools to represent and manipulate data efficiently.
- **Calculus and Optimization** (Chapters on gradients, Hessians, Taylor approximations, and convexity) are essential for understanding how machine learning models are trained through iterative optimization processes.
- **Probability and Statistics** (Chapters on probability basics, random variables, covariance, and estimation) are fundamental for modeling uncertainty, interpreting results, and developing robust algorithms.

Throughout the chapters, each mathematical topic is motivated by practical machine learning examples implemented in Python. By seeing theory in action—such as predicting cancer diagnoses, analyzing weather patterns, understanding genetic data, or recognizing handwritten digits—you'll gain deeper intuition and appreciation for these mathematical tools.

[Jupyter Book](https://jupyterbook.org/) allows us to place the theory directly in the context of practical applications. Interactive Jupyter Notebooks illustrate concepts clearly and concretely, allowing you to view the code, modify it, and explore these concepts on your own.

We invite you to explore the table of contents below to discover the mathematical foundations that will empower you to master machine learning and data science:

```{tableofcontents}
```d