# AI-For-Medicine-Specialization
This is a Coursera provided Specialization which has three courses: 1) AI for Medical Diagnosis, 2) AI for Medical Prognosis and 3) AI for Medical Treatment. 
I have completed all the three courses with full determination and focus. Therefore, I have managed to learn all the sophisticated stuffs in one month duration. 
I am providing here all the key topics with basic explainations in a week by week basis. This will give an overall idea about the course contents and the purpose for taking this Specialization. 

## AI for Medical Diagnosis                      
**Week 1**
- 3 key challenges in medical image classification: 
1. Class imbalance: 
Solve: (1) Use weighted loss. here, rare class loss is multiplied by a bigger number for getting similar level loss as like the more common class. (2) Resampling datasets can solve.
2. Multi-task 
3. Data Size: solution: (1) transfer learning (2) Augmentation: check for: if augmented data resembles real world scenerio and if the data holds its same label as original.
- Medical image datasets typically have 10 thousand to 100 thousand examples.

- Model testing:: 3 key challenges: 
1. patient overlaping( independant dataset): solution: split by patient. in traditional splitting approach, same patient xray image may found in different places which creates problems.
2. set sampling: typically, test set size is 100 for medical image. when sampling, look for atleast x% of samples of the minority class. because validation and test set should have the same type distribution, manually sample validation set before training. training set will have class imbalance, but solution exists :).
3. ground truth : inter-observer disagreement happens between specialists. Solution: Consensus voting
- Why can't we use the same generator in validation and test data as for the training data? 
  - Knowing the average per batch of test data would effectively give our model an advantage. The model should not have any information about the test data.
  - What we need to do is normalize incoming test data using the statistics computed from the training set. There is one technical note. Ideally, we would want to compute our sample mean and standard deviation using the entire training set. However, since this is extremely large, that would be very time consuming. In the interest of time, we'll take a random sample of the dataset and calcualte the sample mean and sample standard deviation.
  
**Week 2**
- Model evaluation metrics:
  - Sensitivity: Ratio of true positive prediction with respect to positive ground truth. sensitivity is how well the model predicts actual positive cases as positive.
  - Specificity: ratio of true negative prediction with resoect to negative ground truth.
  - PPV: Ratio of true positive prediction with respect to total positive predicted. 
  - NPV: Ratio of true negative prediction with respect to total negative predicted.
  - TruePos=Sensitivity×Prevalence
  - PPV= (sensitivity×prevalence) / (sensitivity×prevalence+(1−specificity)×(1−prevalence)) 
- Threshold determines the metrics like: if t =1, then specificity is 1 and sensitivity is 0 and if t =0, then vice versa.
- Note that specificity and sensitivity do not depend on the prevalence of the positive class in the dataset. This is because the statistics are only computed within people of the same class.

**Week 3**
- MRI Data Segmentation
  - 2 types segmentation: 2D and 3D. better: 3D. one MRI is subdivided into many small 3D objects, where each unit called voxel 
  - U-Net model for segmentation and
  - Soft Dice loss for loss function. Soft dice works well for class imbalance. Here brain tumor region is very small so this loss function will work well. It computes the overlap between prediction and Ground truth. EQ: (sum(pi*gi)) / (sum(pi^2) + sum(gi^2)), here, pi= prediction, gi = ground truth (both are 1D column for example purpose) actual MRI data is multi-dimensional


## AI for Medical Prognosis        
**Week 1**  
- Medical Prognosis is about predicting future risk of a disease. Logistic regression model is used for this binary classification of risk or not (0 or 1).
- In diabetic retinopathy problem, train CSV data is skewed in right, which should be removed as the models assume the data to be normally distributed with no skewness. One way to remove the skewness is by applying the log function to the data.
- For evaluating a model, use C-Index. The c-index measures the discriminatory power of a risk score. Intuitively, a higher c-index indicates that the model's prediction is in agreement with the actual outcomes of a pair of patients.
  - To get the predicted probabilities, we use the predict_proba method. it is the Risk Score. then calculate C-Index and look how good the model works
  - plot the coefficients to see which variables (patient features) are having the most effect. You can access the model coefficients by using model.coef_
  - with interaction terms added to both train and test data, C-Index is better than previous.   
  
**Week 2**
- decision tree can model non-linear associations. Decision trees can only draw vertical or horizontal decision boundaries. 
  - Example: in X axis: age and Y axis: BP. Blue dots for alive patients and red dots for dead patients. find a variable and its value to draw a line by which the points are best divided. then repeat this until all points are mostly divided. then count the percentage of red points on each area and detect high risk area or low by applying a threshold. 
  - Fix overfitting:: two methods: decreasing max_depth (which is the count for generating boundary lines) or use Random Forest Tree. Random forest gives higher test accuracy than one single tree. (Ensemble Learning)
- Missing Data::
  - One way to solve is to exclude the rows of missing data from both training and test sets(after train test split). But this creates a problem as we don't know how the data is missed and there may create biases that will reduce accuracy in new test data. 
- Three ways of data missing: Missing at completely random (no biases), missing at random and missing at not random
  - another way to solve is Imputation: two types: mean imputation and regression imputation. Better is the second one because this maintains the relationship between missing and non-missing variables by fitting a linear model on these data. 
- The approach we implement to tune the hyperparameters is known as a grid search::: We define a set of possible values for each of the target hyperparameters.
A model is trained and evaluated for every possible combination of hyperparameters. The best performing set of hyperparameters is returned.




