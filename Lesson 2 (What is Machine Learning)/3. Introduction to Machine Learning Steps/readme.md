![steps in Machine Learning Process](https://i0.wp.com/techbusinessguide.com/wp-content/uploads/2019/05/machine-learning-steps-process-flow.jpg?resize=640,424&ssl=1)

 1. Define the Problem 
 2. Build the Dataset 
 3. Train the model 
 4. Evaluate the model 
 5. Inference(Use the model)

#  Step 1 - Define the Model
## How do You Start a Machine Learning Task?

-   **_Define a very specific task._**
    -   Think back to the snow cone sales example. Now imagine that you own a frozen treats store and you sell snow cones along with many other products. You wonder, "‘How do I increase sales?" It's a valid question, but it's the  **opposite**  of a very specific task. The following examples demonstrate how a machine learning practitioner might attempt to answer that question.
        -   “Does adding a $1.00 charge for sprinkles on a hot fudge sundae increase the sales of hot fudge sundaes?”
        -   “Does adding a $0.50 charge for organic flavors in your snow cone increase the sales of snow cones?”
-   **_Identify the machine learning task we might use to solve this problem._**
    -   This helps you better understand the data you need for a project.

  

## What is a Machine Learning Task?

All model training algorithms, and the models themselves, take data as their input. Their outputs can be very different and are classified into a few different groups based on the  _task_ they are designed to solve. Often, we use the kind of data required to train a model as part of defining a machine learning task.

In this lesson, we will focus on two common machine learning tasks:

-   **Supervised** learning
-   **Unsupervised** learning

  

## Supervised and Unsupervised Learning

The presence or absence of labeling in your data is often used to identify a machine learning task.
![Machine learning classification](https://www.edureka.co/blog/wp-content/uploads/2018/12/Classification-Machine-Learning-Algorithms-Edureka.png)
### Supervised tasks

A task is _supervised_  if you are using labeled data. We use the term  _labeled_  to refer to data that already contains the solutions, called  _labels_.

> For example: Predicting the number of snow cones sold based on the temperatures is an example of supervised learning
### Unsupervised tasks

A task is considered to be  _unsupervised_ if you are using _unlabeled data_. This means you don't need to provide the model with any kind of label or solution while the model is being trained.

**How do we classify tasks when we don't have a label?**

Unsupervised learning involves using data that doesn't have a label. One common task is called  **clustering**. Clustering helps to determine if there are any naturally occurring groupings in the data.

Let's look at an example of how clustering in unlabeled data works.

  

**Identifying book micro-genres with unsupervised learning**

Imagine that you work for a company that recommends books to readers.

_The assumption_: You are fairly confident that micro-genres exist, and that there is one called  _Teen Vampire Romance_. Because you don’t know which micro-genres exist, you can't use  **supervised learning**  techniques.

This is where the  **unsupervised learning**  clustering technique might be able to detect some groupings in the data. The words and phrases used in the book description might provide some guidance on a book's micro-genre.

## Further Classifying by using Label Types
Initially, we divided tasks based on the presence or absence of labeled data while training our model. Often, tasks are further defined by the type of label which is present.

In  **supervised** learning, there are two main identifiers you will see in machine learning:

-   A  **categorical** label _has a_ discrete _set of possible values. In a machine learning problem in which you want to identify the type of flower based on a picture, you would train your model using images that have been labeled with the categories of flower you would want to identify. Furthermore, when you work with categorical labels, you often carry out_ classification tasks*, which are part of the supervised learning family.
-   A  **continuous** (regression) label _does not have a discrete set of possible values, which often means you are working with numerical data. In the snow cone sales example, we are trying to predict the_ number* of snow cones sold. Here, our label is a number that could, in theory, be any value.

In unsupervised learning,  **clustering** is just one example. There are many other options, such as deep learning.

## Terminology

-   **Clustering**. Unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.
-   A  **_categorical label_**  has a discrete set of possible values, such as "is a cat" and "is not a cat."
-   A  **continuous (regression) label**  does not have a discrete set of possible values, which means possibly an unlimited number of possibilities.
-   **Discrete**: A term taken from statistics referring to an outcome taking on only a finite number of values (such as days of the week).
-   A  **label**  refers to data that already contains the solution.
-   Using  **unlabeled** data means you don't need to provide the model with any kind of label or solution while the model is being trained.

## Additional Reading

-   The  [AWS Machine Learning blog](https://aws.amazon.com/blogs/machine-learning/)  is a great resource for learning more about projects in machine learning.
-   You can use Amazon SageMaker  [to calculate new stats in Major League Baseball](https://aws.amazon.com/blogs/machine-learning/calculating-new-stats-in-major-league-baseball-with-amazon-sagemaker/).
-   You can also find an article on  [Flagging suspicious healthcare claims with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/flagging-suspicious-healthcare-claims-with-amazon-sagemaker/)  on the AWS Machine Learning blog.
-   What [kinds of questions and problems](https://docs.aws.amazon.com/machine-learning/latest/dg/machine-learning-problems-in-amazon-machine-learning.html)  are good for machine learning?

# Step 2: Build a Dataset
The next step in the machine learning process is to build a dataset that can be used to solve your machine learning-based problem. Understanding the data needed helps you select better models and algorithms so you can build more effective solutions.

  

### The most important step of the machine learning process

Working with data is perhaps the most overlooked—yet most important—step of the machine learning process. In 2017, an O’Reilly study showed that machine learning practitioners spend 80% of their time working with their data.

## The Four Aspects of Working with Data
You can take an entire class just on working with, understanding, and processing data for machine learning applications. Good, high-quality data is essential for any kind of machine learning project. Let's explore some of the common aspects of working with data.

### Data collection

Data collection can be as straightforward as running the appropriate SQL queries or as complicated as building custom web scraper applications to collect data for your project. You might even have to run a model over your data to generate needed labels. Here is the fundamental question:

> Does the data you've collected match the machine learning task and problem you have defined?

### Data inspection

The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:

-   Outliers
-   Missing or incomplete values
-   Data that needs to be transformed or preprocessed so it's in the correct format to be used by your model

  

### Summary statistics

Models can assume how your data is structured.

Now that you have some data in hand it is a good best practice to check that your data is in line with the underlying assumptions of your chosen machine learning model.

With many statistical tools, you can calculate things like the mean, inner-quartile range (IQR), and standard deviation. These tools can give you insight into the  _scope_,  _scale_, and  _shape_ of the dataset.

### Data visualization

You can use data visualization to see outliers and trends in your data and to help stakeholders understand your data.

## Terminology

-   _Impute_ is a common term referring to different statistical tools which can be used to calculate missing values from your dataset.
-   _Outliers_ are data points that are significantly different from others in the same sample.

## Additional reading

-   In machine learning, you use several statistical-based tools to better understand your data. The  `sklearn`  library has many examples and tutorials, such as this example demonstrating  [outlier detection on a real dataset](https://sklearn.org/auto_examples/applications/plot_outlier_detection_housing.html#sphx-glr-auto-examples-applications-plot-outlier-detection-housing-py).
