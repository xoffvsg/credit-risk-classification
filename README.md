# credit-risk-classification
Module 20 Challenge


## Overview of the Analysis

The purpose of this analysis is to allow a financial company doing peer-to-peer lending services to build a predictive model to identify risky borrowers. The historical data available for the analysis consists of 77536 data points with 6 features (loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, and total_debt) and 1 label (loan_status). <br>
Out of these 77536 loans, 75036 are in good standing and 2500 have defaulted.<br><br>

In order to develop and validate the model, we randomly selected 75% of the dataset for training (58152 entries) and 25% for testing (19384 entries). We stratified the selection to make sure that 25% of defaulted loans are present in both the training and testing subsets.<br><br>

We chose to use a supervised machine learning based on logistic regression to build the model. Once the model was instantiated, we trained it by fitting the training data (features vs known labels). We then compared the predicated outcomes with the actual labels by applying the model to the test data. We then assessed the results by analyzing the confusion matrix and the classification report.<br><br>

Because of the low representation of the defaulted loans in the dataset (3.2% of the total), we investigated how to avoid a model bias by oversampling the number of defaulted loans in the training dataset. Following this resampling, the logistic regression model is fitted again on a perfectly balanced dataset of 56277 healthy loans and 56277 defaulted loans. The results are then compared with the ones obtained previously.


## Results

* Machine Learning Model 1: Logistic Regression
  * Description of Model 1:
    * Precision scores: 
      * 100% of the loans predicted to be healthy were actually healthy
      * 87% of the loans predicted to be risky actually defaulted
    * Recall scores:
      * out of all the healthy loans, the model predicted correctly 100% of them
      * out of all the delinquent loans, the model predicted correctly 89% of them
  * F1-score: The weighted harmonic mean of precision and recall are respectively  1.00 and 0.88 for the healthy and the high-risk loans (the closer to 1 the better)
  * Accuracy:
    * The model accuracy is 0.99
    * The balanced accuracy score of the model is 0.94

<br><br>


* Machine Learning Model 2: Logistic Regression after oversampling (rebalancing)
  * Description of Model 2:
    * Precision scores: 
      * 100% of the loans predicted to be healthy were actually healthy
      * 87% of the loans predicted to be risky actually defaulted
    * Recall scores:
      * out of all the healthy loans, the model predicted correctly 100% of them
      * out of all delinquent loans, the model predicted correctly 100% of them (actually 99.98968%)
    * F1-score: The weighted harmonic mean of precision and recall are respectively  1.00 and 0.93 for the healthy and the high-risk loans (the closer to 1 the better)
    * Accuracy:
      * The model accuracy is 0.99
      * The balanced accuracy score of the model is 1.00 (after rounding to two significant digits)

## Summary

In conclusion, the model using the oversampling prior to the logistic regression fitting does a better job at almost eliminating the risk of a false negative, which would correspond to a predicted no-risk loan that eventually turns out to be delinquent. A false negative has a more direct impact on the financial institution making the loan than a false positive (a loan declined to a borrower who would have repaid as agreed). The Model 2 would have only missed to detect two bad loans out of 19384 loans compared to the 67 bad loans missed by Model 1.<br>
The direct comparison between the two models illustrates the risk of poorly predicting one of the classes when the training data is very imbalanced.
