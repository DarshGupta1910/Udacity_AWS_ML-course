Nearly all tasks solved with machine learning involve three primary components:

-   A machine learning model
-   A model training algorithm
-   A model inference algorithm

## Clay Analogy for Machine Learning

You can understand the relationships between these components by imagining the stages of crafting a teapot from a lump of clay.

1.  First, you start with a block of raw clay. At this stage, the clay can be molded into many different forms and be used to serve many different purposes. You decide to use this lump of clay to make a teapot.
2.  So how do you create this teapot? You inspect and analyze the raw clay and decide how to change it to make it look more like the teapot you have in mind.
3.  Next, you mold the clay to make it look more like the teapot that is your goal.

Congratulations! You've completed your teapot. You've inspected the materials, evaluated how to change them to reach your goal, and made the changes, and the teapot is now ready for your enjoyment.

----------

## What are machine learning models?

A machine learning model, like a piece of clay, can be molded into many different forms and serve many different purposes. A more technical definition would be that a machine learning model is a block of code or framework that can be modified to solve different but related problems based on the data provided.

**Important**

> A model is an extremely generic program(or block of code), made specific by the data used to train it. It is used to solve different problems.

All machine learning models are categorized  as either **supervised** or **unsupervised**. If the model is a supervised model, it’s then sub-categorized as either a **regression** or **classification** model <br>
[All Machine Learning Models](https://towardsdatascience.com/all-machine-learning-models-explained-in-6-minutes-9fe30ff6776a)

## Model Training 
### How are model training algorithms used to train a model?

In the preceding section, we talked about two key pieces of information: a model and data. In this section, we show you how those two pieces of information are used to create a trained model. This process is called  _model training_.

  

### Model training algorithms work through an interactive process

Let's revisit our clay teapot analogy. We've gotten our piece of clay, and now we want to make a teapot. Let's look at the algorithm for molding clay and how it resembles a machine learning algorithm:

-   **Think about the changes that need to be made.**  The first thing you would do is inspect the raw clay and think about what changes can be made to make it look more like a teapot. Similarly, a model training algorithm uses the model to process data and then compares the results against some end goal, such as our clay teapot.
-   **Make those changes**. Now, you mold the clay to make it look more like a teapot. Similarly, a model training algorithm gently nudges specific parts of the model in a direction that brings the model closer to achieving the goal.
-   **Repeat.**  By iterating over these steps over and over, you get closer and closer to what you want until you determine that you’re close enough that you can stop.
## Model Inference: Using Your Trained Model

Now you have our completed teapot. You inspected the clay, evaluated the changes that needed to be made, and made them, and now the teapot is ready for you to use. Enjoy your tea!

_So what does this mean from a machine learning perspective?_  We are ready to use the model inference algorithm to generate predictions using the trained model. This process is often referred to as  **model inference.**

## Terminology

A  **model** is an extremely generic program, made specific by the data used to train it.

**Model training algorithms**  work through an interactive process where the current model iteration is analyzed to determine what changes can be made to get closer to the goal. Those changes are made and the iteration continues until the model is evaluated to meet the goals.

**Model inference**  is when the trained model is used to generate predictions.
