# Machine Learning Guide

This guide is intended for anyone having background in programming and maths. There is no specific order to follow, but a classic path would be from top to bottom. If you don't like reading books, skip it, if you don't want to follow online courses, you can skip it as well. Please don't skip both reading books and following online courses. There is not a single way to become a machine learning expert and with motivation, you can absolutely achieve it. The guide is designed to complete in 4 weeks but it can take more/less time.

```N.B: Members having access to this repo is not allowed to share this resource outside of NSL without prior notification. And please use this resource a study reference only. Feel free to contact the contributor by commenting if you have further query.```

## Table of Contents

- [Machine Learning Roadmap](#machine-learning-roadmap)
- [Tools Used Here](#tools-used-here)
- [Follow Free Online Courses](#follow-free-online-courses)
- [Read Books](#read-books)
- **[Week 01: ML Intro and Supervised Learning (Linear Regression, Logistic Regression)](#week-01-ml-intro-and-supervised-learning-linear-regression-logistic-regression)**
- **[Week 02: Supervised Learning (Support Vector Machines)](#week-02-supervised-learning-support-vector-machines)**
- **[Week 03 | Part 1: Tuning and Performance Metrics of ML Model](#week-03--part-1-tuning-and-performance-metrics-of-ml-model)**
- **[Week 03 | Part 2: Ensemble Learning](#week-03--part-2-ensemble-learning)**
- **[Week 04: Unsupervised Learning](#week-04-unsupervised-learning)**
- [Be a Kaggler](#be-a-kaggler)

## Machine Learning Roadmap

[![Machine Learning Roadmap Overview](./images/ml-roadmap-overview.png)](https://whimsical.com/machine-learning-roadmap-2020-CA7f3ykvXpnJ9Az32vYXva)

## Tools Used Here

- **[Scikit-Learn](https://scikit-learn.org/stable/)**: Scikit-learn is probably the most useful library for traditional machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
- **[NumPy](https://numpy.org)**: NumPy is an open-source numerical Python library. NumPy contains a multi-dimensional array and matrix data structures. It can be utilised to perform a number of mathematical operations on arrays such as trigonometric, statistical, and algebraic routines.
- **[SciPy](https://www.scipy.org/)**: SciPy is an open-source Python library which is used to solve scientific and mathematical problems. It is built on the NumPy extension and allows the user to manipulate and visualize data with a wide range of high-level commands.
- **[Matplotlib](https://matplotlib.org/)**: Matplotlib is a cross-platform, data visualization and graphical plotting library for Python and its numerical extension NumPy. As such, it offers a viable open source alternative to MATLAB.
- **[Pandas](https://pandas.pydata.org/)**: Pandas is one of the tools in Machine Learning which is used for data cleaning and analysis. It has features which are used for exploring, cleaning, transforming and visualizing from data. Pandas is an open-source python package built on top of Numpy developed by Wes McKinney.

## Follow Free Online Courses

Here is a list of awesome courses that you should definitely follow and are 100% free. You can pick one, two or all considering your requirements but you should follow at least one.

1. Condensed Overview on Machine Learning: **[Machine Learning Crash Course by Google](https://developers.google.com/machine-learning/crash-course)**.
2. Machine Learning course by Stanford [CS229: Machine Learning](http://cs229.stanford.edu/syllabus-spring2020.html)
3. Academic Course on Machine Learning: [Machine Learning by Andrew NG - YouTube Playlist (Stanford)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU).
4. Machine Learning Algorithms Explained: [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer).

## Read Books

Here is the great book to read for the Machine Learning Engineers. This books consists of two parts [Part 1: Machine Learning, Part 2: Deep Learning]. You should follow the Part 1 for this guide.

**[Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems - by Aurelien Geron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)**.

Download the book from [this link](./books/Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_TensorFlow_2nd_edition.pdf). **The pdf book is only for study purpose not for distribution or sale**.

For the code practice, please follow author's [github repository](https://github.com/ageron/handson-ml2).

If you prefer to study in Bangla to introduce with Machine Learning as beginner,

- [Bangla gitbook on Machine Learning](https://raqueeb.gitbook.io/scikit-learn/)
- [বাংলায় মেশিন লার্নিং](https://ml.howtocode.dev/)


## Week 01: ML Intro and Supervised Learning (Linear Regression, Logistic Regression)
### Requied Topics:
1. Definition of Machine Learning, Supervised Learning and Unsupervised Learning: [ML Intro in Bangla](https://ml.howtocode.dev/) and [Lecture 1.1 - 1.3 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
2. Simple Linear Regression: [Lecture 2.1 - 2.8 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
3. Linear Algebra Review: [Reading](https://towardsdatascience.com/linear-algebra-for-machine-learning-22f1d8aea83c) and [Lecture 3.1 - 3.6 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
4. Linear Regression with Multiple Variable: [Lecture 4.1 - 4.7 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
5. Logistic Regression: [Lecture 6.1 - 6.7 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).

### Recommended Videos:
1. Definition of Machine Learning, Supervised Learning and Unsupervised Learning: [1:15 hr lecture](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU).
2. Linear Regression: [1:18 hr lecture](https://www.youtube.com/watch?v=4b4MUYve_U8&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=2).
3. Logistic Regression: [1:19 hr lecture](https://www.youtube.com/watch?v=het9HFqo1TQ&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=3).

### Code Examples:
1. [Data Preprocessing for Machine Learning](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/02_Data_Preprocessing).
2. [Data Visualization](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/03_Data_Visualization).
3. [Simple Linear Regression](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/04_Simple%20_Linear_Regression).
4. [Multiple Linear Regression](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/05%20Multiple%20Linear%20Regression).
5. [Polynomial Regression](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/06_Polynomial_Regression).
6. [Logistic Regression](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/09_Logistic_Regression)

## Week 02: Supervised Learning (Support Vector Machines)
1. [Support Vector Machines](https://www.youtube.com/watch?v=1NxnPkZM9bc).

### Recommended Videos:
1. Support Vector Machines Explained: [Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE), [Part 2](https://www.youtube.com/watch?v=Toet3EiSFcM), [Part 3](https://www.youtube.com/watch?v=Qc5IyLW_hns).
2. [Support Vector Machines in Python](https://www.youtube.com/watch?v=8A7L0GsBiLQ).
3. Support Vector Machines: [Lecture 12.1 - 12.6 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).

### Code Examples:
1. [Naive Bayes](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/12_Naive_Bayes)
2. [K-Nearest Neighbor](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/10_K_Nearest_Neighbors)
3. [Support Vector Machines](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/07_Support_Vector_Regression).


## Week 03 | Part 1: Tuning and Performance Metrics of ML Model
### Requied Topics:
1. Regularization in Traditional ML | Part 1: [Lecture 7.1 - 7.4 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
2. Regularization in Traditional ML | Part 2: [Lecture 10.1 - 10.7 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
3. Performance Metrics: [Lecture 11.1 - 11.5 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
### Code Examples:
1. [Regularization in Traditional ML](http://ethen8181.github.io/machine-learning/regularization/regularization.html)

## Week 03 | Part 2: Ensemble Learning

### Pre-Requisites of Ensemble Learning:
1. Decision Tree: [Decision Tree Explained](https://www.youtube.com/watch?v=7VeUPuFGJHk), [Decision Tree Summary](https://www.youtube.com/watch?v=tNa99PG8hR8).
2. Regression Tree: [Regression Tree Explained](https://www.youtube.com/watch?v=g9c66TUylZ4).
3. Random Forest: [Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ), [Part 2](https://www.youtube.com/watch?v=nyxTdL_4Q-Q).

### Requied Topics:
1. Ensemble Learning: [Bagging, Boosting and Stacking Explained](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205).
2. AdaBoost: [AdaBoost Explained](https://www.youtube.com/watch?v=LsK-xG1cLYA).
3. Gradient Boost: [Part 1](https://www.youtube.com/watch?v=3CC4N4z3GJc), [Part2](https://www.youtube.com/watch?v=2xudPOBz-vs), [Part3](https://www.youtube.com/watch?v=jxuNLH5dXCs), [Part4](https://www.youtube.com/watch?v=StWY5QWMXCw).
4. XGBoost: [Part 1](https://www.youtube.com/watch?v=OtD8wVaFm6E), [Part 2](https://www.youtube.com/watch?v=8b1JEDvenQU), [Part 3](https://www.youtube.com/watch?v=ZVFeW798-2I), [Part 4](https://www.youtube.com/watch?v=oRrKeUCEbq8).

### Recommended Video:
[Decision Trees and Ensemble Methods](https://www.youtube.com/watch?v=wr9gUr-eWdA&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=10)

### Code Examples:
1. Decision Tree and Random Forest: [Intro](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/08_Decision_Tree%26_Random_Forest)
2. Decision Tree: [Decision Tree Classifier](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/13_Decision_Trees)
3. Random Forest: [Random Forest Classification](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/14_Random_Forest_Classification)
4. Adaboost: [AdaBoost Ensemble in Python](https://machinelearningmastery.com/adaboost-ensemble-in-python/)
5. XGBoost: [Using XGBoost in Python](https://www.datacamp.com/community/tutorials/xgboost-in-python)

## Week 04: Unsupervised Learning

### Required Topics:
1. Clustering: [Lecture 13.1 - 13.5 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
2. Dimensionality Reduction: [Lecture 14.1 - 14.7 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
3. Anomaly Detection: [Lecture 15.1 - 15.8 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
4. Recommender Systems: [Lecture 16.1 - 16.6 of this playlist](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN).
5. Association Rule Mining and Apriori Algorithms: [Association Rule Mining and Apriori Algorithms Explained](https://www.youtube.com/watch?v=guVvtZ7ZClw).

### Code Examples:
1. Clustering: [K-Means Clustering](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/15_K_Means_Clustering), [DBSCAN Clustering](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/16_DBSCAN_Clustering), [Hierarchical Clustering](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/17_Hierarchical_Clustering), [Customer Segmentation](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/18_Customer_Segmentation).
2. Dimensionality Reduction: [Principal Component Analysis](https://github.com/mhuzaifadev/Machine_Learning_Algorithms/blob/master/09_Clustering/Principal_Component_Analysis.ipynb).
3. Anomaly Detection: [Credit Card Fraud Detection](https://www.youtube.com/watch?v=frM_7UMD_-A).
4. Recommender Systems: [Hybrid Recommendation Systems using LightFM](https://github.com/mhuzaifadev/mlzero_to_hero/tree/main/29_Hybrid_Recommendation_Systems).
5. Association Rule Mining and Apriori Algorithms: [Implementing Apriori Algorithm in Python](https://www.youtube.com/watch?v=SVM_pX0oTU8).

### Combined Exercises
[Python Exercises based on Andrew Ng's Machine Learning Course.](https://github.com/dibgerge/ml-coursera-python-assignments)

## Be a Kaggler
Try to register in **[Kaggle](https://www.kaggle.com/)** and participate in Machine Learning Competition to get some flavor of Problem Solving on Traditional Machine Learning or read through kaggle notebooks to know the crafts for approaching to solve any Machine Learning problems.

