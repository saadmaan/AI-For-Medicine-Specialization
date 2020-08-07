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

### [Certificate](https://www.coursera.org/account/accomplishments/records/HKMC2CPA8B4V)

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

**Week 3**
- Survival Model:: what is the probability that a patient will survive for greater than some years, this is what we will predict in this model
- As time increases, survival function in a graph where t(time) is in X axis, should never go up, it goes downwards or stays the same

**Week 4**
- Hazard Function: what is their immediate risk of death if they make it to time t
- Survival, hazard and cumulative hazard function are related. One can be derived from other
- Till now, we have calculated these functions for all the populations, which are the same for all patients. But in reality, each patient has different characteristics and for this, we calculate hazard for individual persons. Function:     lambda(individual) = lambda(baseline) * factor 
factor equation is, for example: exp(.08*is_smoker + .01*age)
  - by this, we can compare and rank patients by their risk. 
  - we can compare by taking ratios of two patients. this is called ''Cox proportional Hazard model"
  - generalized form :   lambda(individual) = lambda(baseline) * exp( sum(Beta(i) * X(i)) )
  - exp(Beta(i)) is the factor risk increase for a unit increase of any variable like age or is_smoker. 
- Non-linear risk models can be generated using survival trees. Above function is linear and can not determine non-linear relations. So, a decision tree is made like previous(course 1), though the end result is not anymore classifications but risk values at any time, t. 
  - At first we make different groups by tree model and we determine which group the particular patient falls into. Then, we calculate cumulative hazard for that group of patients.
- Nelson Aalen Estimator is used to calculate cumulative hazard of a population
- Risk models develop using survival data
- "lifelines" is an open-source survival analysis library.
- CoxPHFitter().fit() : this cox proportional hazard model trains on survival data and generates coefficients for each covariates. From this, Hazard Ratio between two different patients can be calculated which can tell who is at more risk. 
  - Then we can evaluate model by harrell's C-index
  - If we use Random Survival forests, then performance will be better. Here, instead of encoding our categories as binary features, we can use the original dataframe since trees deal well with raw categorical data
  -  random survival forests come with their own built in variable importance feature. The method is referred to as VIMP, and for the purpose of this section we should just know that higher absolute value of the VIMP means that the variable generally has a larger effect on the model outcome.  

### [Certificate](https://www.coursera.org/account/accomplishments/records/6MXFK42VT6GW)

# AI For Medical Treatment    
**Week 1**
- machine learning techniques for predicting effects of a treatment
  - AR(treatment) - AR(control) = ARR  [ARR= Absolute Risk Reduction]
  - RCT (Randomized Control Trial) is used for the above testing of the quality of new treatment. It is nothing but random assignment of patients into two or more groups. 
  - Using P values, we can convey the statistical significance of our result 
  - More interpretable definition for measuring the effect of treatment is NNT(Number needed for treatment). NNT =33 means that if 33 people get trtmnt, then one person can be saved. NNT = 1/ ARR 
  - NEYMAN RUBIN Causal Model (Estimation of how good a treatment works in patients in comparison to patients without treatment): we calculate average treatment effect(ATE) by: Expectation(Y[i]1 - Y[i]0) = Expectation(Y[i] | W=1) - Expectation(Y[i] | W=0); W =1 means with treatment. For this, data must be from RCT
  - ATE = - ARR
  - Two Tree Method / T-Learner
Here, we calculate CATE(Conditional ATE) by estimating the result by two different prognostic models for W=1 and W=2 from RCT data. It can be tree models or linear. 
  - S-learner:: either this or T-learner can be used. S-learner's limitation is that it might not use the treatment feature for learning and thus, there will be no difference in predictions for both with and without treatment. 
  - T-learner limitation is that data is divided into two and thus model gets fewer data to learn than S-learner 
- Interpretation of Logistic Regression Model:::
  - by Odds: p/(1-p) where, p is the prediction of an event(such as death) of the model. 
  - the log of odds is defined as the logit function: logit used because of simplicity rather than exponents
  - Interpret Odds Ratio:
  - The features with a negative coefficient has an Odds Ratio that is smaller than 1
and features with non-negative coefficients have ORs greater than 1.
However, now that you have calculated the Odds Ratio of 0.66 for TRTMT, you can interpret this value:
If the patient does not receive treatment, and let's assume that their odds of dying is some value like 2.
If the patient does receive treatment, then the patient's odds of dying is  0.66×2 .
In other words, the odds of dying if given treatment is 0.66 times the odds of dying if not receiving treatment.

- **Programming Assignment**
- Estimating Treatment Effect Using Machine Learning::::
  - data analysis and visualization: check shape and see by .head
  - find what the treatment probability is. Here, trtmnt portion is 49%. As this is RCT data, column is chosen randomly and so both terms are nearly equal
  - then find the death rate for treatment and non-treatment. Here, we see the treatment has positive outcome
  - generally train test splitting for logistic reg: 
    - data = data.dropna(axis=0)
    - y = data.outcome
    - notice we are dropping a column here. Now our total columns will be 1 less than before
    - X = data.drop('outcome', axis=1) 
    - X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
  - Logistic regression is: sigma(theta(t)*X(i)). Another way to look at logistic regresion is as a linear model for the "logit" function, or "log odds": logit(p) = log(p/(1-p))=theta(t)*X(i)
  - Fit the model
  - Extract the constant treatment effect by calculating odds ratio. Odds ratio = Odds(treatment) / Odds(Control) = np.exp(theta(treatment)). As we just calculate treatment effect, we keep all other features' values constant. 
  - Odds ratio smaller than 1 means treatment reduces the death probability. if OR = 0.75, we can tell that the treatment reduces the odds of death by  (1−OddsRatio)=1−0.75=0.25 , or about 25%.
  - then calculate ARR by OR and P(baseline)
  - The logistic regression model assumes that treatment has a constant effect in terms of odds ratio and is independent of other covariates. However, this does not mean that absolute risk reduction is necessarily constant for any baseline risk. It is always important to remember that baseline risk has a natural effect on absolute risk reduction.
  - ARR may depend on more than simply the baseline risk.
- Evaluating Model by evaluation metric::::
  - The fundamental problem is that for each person, you can only observe either their treatment outcome or their baseline outcome. So, you can't determine what their actual risk reduction was. 
  -  Predicted Risk Reduction is needed to compute C-statistic for benefit. The easiest way to do this in general is to create a version of the data where the treatment variable is False and a version where it is True. Then take the difference: pred_RR=p(control)−p(treatment)
  - We can see that even though the model accurately predicts overall risk (regular c-index), it does not necessarily do a great job predicting benefit from treatment (c-for-benefit). 
  - We can also visually assess the discriminative ability of the model by checking if the people it thinks benefit the most from treatment empirically (actually) experience a benefit. By plotting empirical vs predicted risk reduction, we can see that the model performs poorly.
- T-Learner:::
  - because, we see that Logistic Regression is giving far less good predictions. We use T-Learner. 
  - In order to tune your two models, you will use grid search to find the desired parameters. To feed the hyperparameters into a random forest model, you can use a dictionary, so that you do not need to hard code the parameter names.

**Week 2**
- Medical Question answering:::
  - BERT/ ELMo / XLNET are some models for this task. We will see BERT.
Challenge is the last step of answer extraction which is to define the shortest segment of the passage that answers the question.
  - input words by tokens into some layer of transformer blocks of BERT model, and then produce a list of vectors for all the words each having 768 dimensions. By these, we can measure word similarity by measuring distance between them in the vector space. For visualizing, we can use t-SNE visualization to reduce vectors into 2D shape and then view the distances among them. 
  - One challenge is differentiating the same words with different meanings. Word2Vec and Glove include Non-Contextual word representation. But, ELMo, BERT contains Contextual repr. 
  - BERT model learns two vectors (S & E) for every input word or tokens. then passing S and E scores in a grid where S in rows and E in columns, we calculate the highest score by (S+E) scores for each of the grid cells. For which the words get the highest score, are the predicted start and end word of the answer of the given question. 
  -  Automatic Label extraction from radiology Reports (Ex: for Chest X-ray image classification, we need to extract labels from reports) ::: If labels exist for all the images reports, then supervised learning is used like BERT. But the main challenge is, we don't have labels for reports. 
  - for extracting labels, two step process: is observation mentioned |||  is observation present or absent
  - we can use terminologies/thesaurus/vocabulary like SNOMED CT, which contains synonyms and subtypes (Is-a relationship) for each of the 300,000 concepts like Common Cold, Pneumonia etc. For the Is-a, no labeled data is needed for supervised learning. But lots of manual work needed to refine this rule-based approach for which is working and which isn't. So, this is time consuming and also requires expertise. 
  - now, for the second term, is absent or present, we can choose different rules like regex rules, dependency parser rules (more sophisticated). Ex: No XXX or Pneumonia / Without Pneumonia etc. patterns dictate the absence of that disease. 
- Cleaning text::::
  - 'Lookbehind' assertion:: If you want to match the single letter but not include it in the returned match, you can use the lookbehind assertion: (?<=...)
    - m = re.search(pattern = '(?<=[a-zA-Z])123', string = "99C12399") returns 123
  - You can do the reverse by using the lookahead assertion. (?=...). 
- BioC module transforms clinical data into a standard format that allows you to apply other, more specialized libraries. We will use a BioCCollection object, which represents a collection of documents for a project. The collection might be an entire corpus, or a partial one.
  - NegBio, a tool that distinguishes negative or uncertain findings in radiology reports. It accomplishes this by using patterns on universal dependencies. We will use NegBioSSplitter object to split your text into sentences. In order to do this, you'll first need to convert your text into a format that BioC supports. For this you'll use the text2bioc() function, which transforms the text into a BioC XML file.
- Evaluating on multiple disease categories:::: Macro-average or Micro-average( calculating precision, recall and F1 score)
- when extracting labels, checking for negative words by creating a negative word list and then mentioning those diseases as absent, increase F1 score little bit. But, a more improved way is dependency parsing, where negative mention of diseases can be extracted from the structure of the sentence.
- Implementations of dependency parsers are very complex, but luckily there are some great off-the-shelf tools to do this:::
  - One example is NegBio, a package specifically designed for finding negative and uncertain mentions in X-ray radiology reports.
  - In addition to detecting negations, negbio can be configured to use a dependency parser that has been specifically trained for biomedical text.
  - This increases our performance given that biomedical text is full of acronyms, nomenclature and medical jargon.
- So far we have used a simple rule-based system for predicting answers as we just answer the question of whether a disease is present or not. But for answering variety of questions, we need more advanced system and that is done by a pretrained BERT model, Tokenizer, questions, passage

**WEEK 3**
- feature importance: 
  - (1) Drop column method, which can be computationally expensive as feature increases
  - (2) permutation method; better 
  
### [Certificate](https://www.coursera.org/account/accomplishments/records/ADMKDKF3FQK8)
