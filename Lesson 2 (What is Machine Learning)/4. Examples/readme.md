# Examples 
-   **Supervised learning**
    -   Using machine learning to predict housing prices in a neighborhood based on lot size and number of bedrooms
-   **Unsupervised learning**
    -   Using machine learning to isolate micro-genres of books by analyzing the wording on the back cover description.
-   **Deep neural network**
    -   While this type of task is beyond the scope of this lesson, we wanted to show you the power and versatility of modern machine learning. You will see how it can be used to analyze raw images from lab video footage from security cameras, trying to detect chemical spills.

1. [Hpuse Price Detection Supervised](https://github.com/DarshGupta1910/Udacity_AWS_ML-course/new/main/Lesson%202%20%28What%20is%20Machine%20Learning%29#example-1-house-price-prediction)
2. [Book Genre exploration Unsupervised](https://github.com/DarshGupta1910/Udacity_AWS_ML-course/new/main/Lesson%202%20%28What%20is%20Machine%20Learning%29#example-2---book-genre-exploration)
3. [Spill detection from video Neural Network](https://github.com/DarshGupta1910/Udacity_AWS_ML-course/new/main/Lesson%202%20%28What%20is%20Machine%20Learning%29#example-3---spill-detection-from-video)

# Example 1: House Price Prediction
[House Price Prediction blog](https://towardsdatascience.com/predicting-house-prices-with-machine-learning-62d5bcd0d68f)
House price prediction is one of the most common examples used to introduce machine learning.

Traditionally, real estate appraisers use many quantifiable details about a home (such as number of rooms, lot size, and year of construction) to help them estimate the value of a house.

You detect this relationship and believe that you could use machine learning to predict home prices.

![Machine language to determine house prices](https://video.udacity-data.com/topher/2021/April/608c65ec_house-value/house-value.png)

Machine language models to determine house values

## Step One: Define the Problem

> Can we estimate the price of a house based on lot size or the number of bedrooms?

You access the sale prices for recently sold homes or have them appraised. Since you have this data, this is a  _supervised learning_  task. You want to predict a continuous numeric value, so this task is also a  _regression_  task.

![Supervised learning regression task](https://video.udacity-data.com/topher/2021/April/608c4f9c_supercont/supercont.png)

Regression task

## Step Two: Building a Dataset

-   **Data collection**: You collect numerous examples of homes sold in your neighborhood within the past year, and pay a real estate appraiser to appraise the homes whose selling price is not known.
-   **Data exploration**: You confirm that all of your data is numerical because most machine learning models operate on sequences of numbers. If there is textual data, you need to transform it into numbers. You'll see this in the next example.
-   **Data cleaning**: Look for things such as missing information or outliers, such as the 10-room mansion. Several techniques can be used to handle outliers, but you can also just remove those from your dataset.

![Data cleaning - outlier values](https://video.udacity-data.com/topher/2021/April/608c681c_datatable/datatable.png)

Data cleaning: removing outlier values

-   **Data visualization**: You can plot home values against each of your input variables to look for trends in your data. In the following chart, you see that when lot size increases, the house value increases.

![](https://video.udacity-data.com/topher/2021/April/607593b4_lot-size/lot-size.png)

Regression line of a model

## Step Three: Model Training

Prior to actually training your model, you need to split your data. The standard practice is to put 80% of your dataset into a training dataset and 20% into a test dataset.

  

### Linear model selection

As you see in the preceding chart, when lot size increases, home values increase too. This relationship is simple enough that a linear model can be used to represent this relationship.

A linear model across a single input variable can be represented as a line. It becomes a plane for two variables, and then a hyperplane for more than two variables. The intuition, as a line with a constant slope, doesn't change.

### Using a Python library

[The Python  `scikit-learn`  library](https://scikit-learn.org/stable/)  has tools that can handle the implementation of the model training algorithm for you.

## Step Four: Evaluation

One of the most common evaluation metrics in a regression scenario is called  _root mean square_  or  _RMS_. The math is beyond the scope of this lesson, but RMS can be thought of roughly as the "average error” across your test dataset, so you want this value to be low.

![The math behind RMS](https://video.udacity-data.com/topher/2021/April/6075c540_rms/rms.png)

The math behind RMS

In the following chart, you can see where the data points are in relation to the blue line. You want the data points to be as close to the "average" line as possible, which would mean less net error.

You compute the  _root mean square_  between your model’s prediction for a data point in your test dataset and the true value from your data. This actual calculation is beyond the scope of this lesson, but it's good to understand the process at a high level.

![](https://video.udacity-data.com/topher/2021/April/6075c5c6_rms-chart/rms-chart.png)

## Interpreting Results

In general, as your model improves, you see a better RMS result. You may still not be confident about whether the specific value you’ve computed is good or bad.

Many machine learning engineers manually count how many predictions were off by a threshold (for example, $50,000 in this house pricing problem) to help determine and verify the model's accuracy.

## Step Five: Inference: Try out your model

Now you are ready to put your model into action. As you can see in the following image, this means seeing how well it predicts with new data not seen during model training.

![Testing out the model](https://video.udacity-data.com/topher/2021/April/6075c77d_inf1/inf1.png)

## Terminology

-   **Continuous**: Floating-point values with an infinite range of possible values. The opposite of categorical or discrete values, which take on a limited number of possible values.
-   [**Hyperplane**](https://en.wikipedia.org/wiki/Hyperplane): A mathematical term for a surface that contains more than two planes.
-   [**Plane**](https://en.wikipedia.org/wiki/Plane_(geometry): A mathematical term for a flat surface (like a piece of paper) on which two points can be joined by a straight line.
-   **Regression**: A common task in supervised machine learning.

## Additional reading

The  [Machine Learning Mastery](https://machinelearningmastery.com/)  blog is a fantastic resource for learning more about machine learning. The following example blog posts dive deeper into training regression-based machine learning models.

-   [How to Develop Ridge Regression Models in Python](https://machinelearningmastery.com/ridge-regression-with-python/)  offers another approach to solving the problem in the example from this lesson.
-   Regression is a popular machine learning task, and you can use  [several different model evaluation metrics](https://machinelearningmastery.com/regression-metrics-for-machine-learning/)  with it.

# Example 2 - Book Genre exploration
### Step One: Define the Problem

![Model used to predict micro-genres](https://video.udacity-data.com/topher/2021/April/608c6660_books/books.png)

Model used to predict micro-genres

> Find clusters of similar books based on the presence of common words in the book descriptions.

You do editorial work for a book recommendation company, and you want to write an article on the largest book trends of the year. You believe that a trend called "micro-genres" exists, and you have confidence that you can use the book description text to identify these micro-genres.

By using an unsupervised machine learning technique called  _clustering_, you can test your hypothesis that the book description text can be used to identify these "hidden" micro-genres.

Earlier in this lesson, you were introduced to the idea of unsupervised learning. This machine learning task is especially useful when your data is not labeled.

![Unsupervised learning using clustering](https://video.udacity-data.com/topher/2021/April/608c4fe5_unsuperclust/unsuperclust.png)

Unsupervised learning using clustering

## Step Two: Build your Dataset

To test the hypothesis, you gather book description text for 800 romance books published in the current year.

  

### Data exploration, cleaning and preprocessing

For this project, you believe capitalization and verb tense will not matter, and therefore you remove capitals and convert all verbs to the same tense using a Python library built for processing human language. You also remove punctuation and words you don’t think have useful meaning, like '_a'_  and '_the'_. The machine learning community refers to these words as  _stop words_.

Before you can train the model, you need to do some data preprocessing, called  _data vectorization_, to convert text into numbers.

You transform this book description text into what is called a  `bag of words`representation shown in the following image so that it is understandable by machine learning models.

How the  bag of words representation works is beyond the scope of this course. If you are interested in learning more, see the  **Additional Reading**  section at the bottom of the page.

![bag of words](https://video.udacity-data.com/topher/2021/April/6075cfb1_bag-of-wrods/bag-of-wrods.png)

## Step Three: Train the Model

Now you are ready to train your model.

You pick a common cluster-finding model called  `k-means`. In this model, you can change a model parameter,  `k`, to be equal to how many clusters the model will try to find in your dataset.

Your data is unlabeled: you don't how many microgenres might exist. So you train your model multiple times using different values for  `k`  each time.

What does this even mean? In the following graphs, you can see examples of when  `k=2`  and when  `k=3`.

![kmeans k=2](https://video.udacity-data.com/topher/2021/April/60798990_k2b/k2b.png)  
K=2

![kmeans k=3](https://video.udacity-data.com/topher/2021/April/6079899c_k3b/k3b.png)  
K=3

During the model evaluation phase, you plan on using a metric to find which value for  **k**  is most appropriate.

## Step Four: Model Evaluation

In machine learning, numerous statistical metrics or methods are available to evaluate a model. In this use case, the  _silhouette coefficient_  is a good choice. This metric describes how well your data was clustered by the model. To find the optimal number of clusters, you plot the silhouette coefficient as shown in the following image below. You find the optimal value is when  `k=19`.

![Optimum number (K=19) of  clusters ](https://video.udacity-data.com/topher/2021/April/60899147_k19b/k19b.png)

Optimum number (k=19) of clusters

Often, machine learning practitioners do a manual evaluation of the model's findings.

You find one cluster that contains a large collection of books you can categorize as “paranormal teen romance.” This trend is known in your industry, and therefore you feel somewhat confident in your machine learning approach. You don’t know if every cluster is going to be as cohesive as this, but you decide to use this model to see if you can find anything interesting about which to write an article.

  

## Step Five: Inference (Use the Model)

As you inspect the different clusters found when  `k=19`, you find a surprisingly large cluster of books. Here's an example from fictionalized cluster #7.

![clustered data results for silhouette coefficient](https://video.udacity-data.com/topher/2021/April/6075d66a_silhou/silhou.png)

Clustered data

As you inspect the preceding table, you can see that most of these text snippets are indicating that the characters are in some kind of long-distance relationship. You see a few other self-consistent clusters and feel you now have enough useful data to begin writing an article on unexpected modern romance microgenres.

## Terminology

-   **Bag of words**: A technique used to extract features from the text. It counts how many times a word appears in a document (corpus), and then transforms that information into a dataset.
-   **Data vectorization**: A process that converts non-numeric data into a numerical format so that it can be used by a machine learning model.
-   **Silhouette coefficient**: A score from -1 to 1 describing the clusters found during modeling. A score near zero indicates overlapping clusters, and scores less than zero indicate data points assigned to incorrect clusters. A score approaching 1 indicates successful identification of discrete non-overlapping clusters.
-   **Stop words**: A list of words removed by natural language processing tools when building your dataset. There is no single universal list of stop words used by all-natural language processing tools.

## Additional reading

[Machine Learning Mastery](https://machinelearningmastery.com/)  is a great resource for finding examples of machine learning projects.

-   The  [How to Develop a Deep Learning Bag-of-Words Model for Sentiment Analysis (Text Classification)](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/)  blog post provides an example using a bag of words–based approach pair with a deep learning model.

# example 3 - Spill Detection from Video

## Step One: Defining the Problem

Imagine you run a company that offers specialized on-site janitorial services. A client, an industrial chemical plant, requires a fast response for spills and other health hazards. You realize if you could  _automatically_  detect spills using the plant's surveillance system, you could mobilize your janitorial team faster.

**_Machine learning could be a valuable tool to solve this problem._**

![machine learning for spill detection](https://video.udacity-data.com/topher/2021/April/60871da4_spills/spills.png)

Detecting spills with machine learning

## Step Two: Model Training (and selection)

This task is a supervised classification task, as shown in the following image. As shown in the image above, your goal will be to predict if each image belongs to one of the following classes:

-   **Contains spill**
-   **Does not contain spill**

![Image classification](https://video.udacity-data.com/topher/2021/April/608c502d_superclass/superclass.png)

Image classification

## Step Two: Building a Dataset

-   **Collecting**
    -   Using historical data, as well as safely staged spills, you quickly build a collection of images that contain both spills and non-spills in multiple lighting conditions and environments.
-   **Exploring and cleaning**
    -   You go through all the photos to ensure the spill is clearly in the shot. There are Python tools and other techniques available to improve image quality, which you can use later if you determine a need to iterate.
-   **Data vectorization** (converting to numbers)
    -   Many models require numerical data, so all your image data needs to be transformed into a numerical format. Python tools can help you do this automatically.
    -   In the following image, you can see how each pixel in the image on the left can be represented in the image on the right by a number between 0 and 1, with 0 being completely black and 1 being completely white.

![chemical spill](https://video.udacity-data.com/topher/2021/April/6075d94c_black-spill/black-spill.png)  
Chemical spill image  

![numeric representation of spill](https://video.udacity-data.com/topher/2021/April/6075d93c_spillnumbers/spillnumbers.png)  
Numeric representation of chemical spill image

**Split the data**

-   You split your image data into a training dataset and a test dataset.

## Step Three: Model Training

Traditionally, solving this problem would require hand-engineering features on top of the underlying pixels (for example, locations of prominent edges and corners in the image), and then training a model on these features.

Today, deep neural networks are the most common tool used for solving this kind of problem. Many deep neural network models are structured to learn the features on top of the underlying pixels so you don’t have to learn them. You’ll have a chance to take a deeper look at this in the next lesson, so we’ll keep things high-level for now.

  

### CNN (convolutional neural network)

Neural networks are beyond the scope of this lesson, but you can think of them as a collection of very simple models connected together. These simple models are called  _neurons_, and the connections between these models are trainable model parameters called  _weights._

Convolutional neural networks are a special type of neural network particularly good at processing images.

## Step Four: Model Evaluation

As you saw in the last example, there are many different statistical metrics you can use to evaluate your model. As you gain more experience in machine learning, you will learn how to research which metrics can help you evaluate your model most effectively. Here's a list of common metrics:

  

Accuracy

False positive rate

Precision

Confusion matrix

False negative rate

Recall

F1 Score

Log Loss

ROC curve

Negative predictive value

Specificity

In cases such as this, accuracy might not be the best evaluation mechanism.

**_Why not?_** You realize the model will see the '**Does not contain spill'**  class almost all the time, so any model that just predicts “**no spill**” most of the time will seem pretty accurate.

> > What you really care about is an evaluation tool that rarely misses a real spill.

After doing some internet sleuthing, you realize this is a common problem and that  `Precision`  and  `Recall`  will be effective. You can think of  _precision_  as answering the question, "Of all predictions of a spill, how many were right?" and  _recall_  as answering the question, "Of all actual spills, how many did we detect?"

Manual evaluation plays an important role. You are unsure if your staged spills are sufficiently realistic compared to actual spills. To get a better sense how well your model performs with actual spills, you find additional examples from historical records. This allows you to confirm that your model is performing satisfactorily.

## Step Five: Model Inference

The model can be deployed on a system that enables you to run machine learning workloads such as AWS Panorama.

Thankfully, most of the time, the results will be from the class '**Does not contain spill.'**

![no spill detected](https://video.udacity-data.com/topher/2021/April/60871e9c_nospilled/nospilled.png)

No spill detected

But, when the class '**Contains**  **spill'** is detected, a simple paging system could alert the team to respond.

![spill detected](https://video.udacity-data.com/topher/2021/April/60871e8b_spilled/spilled.png)

Spill detected

## Terminology

**Convolutional neural networks(CNN)**  are a special type of neural network particularly good at processing images.

**Neural networks**: a collection of very simple models connected together.

-   These simple models are called  **neurons**
-   the connections between these models are trainable model parameters called  **weights.**

## Additional reading

As you continue your machine learning journey, you will start to recognize problems that are excellent candidates for machine learning.

The  [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)  is a great resource for finding more examples of machine learning projects.

-   In the  [Protecting people from hazardous areas through virtual boundaries with Computer Vision](https://aws.amazon.com/blogs/machine-learning/protecting-people-through-virtual-boundaries-computer-vision/)  blog post, you can see a more detailed example of the deep learning process described in this lesson.
