# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a Random Forest Classifier aiming to classify whether an individual has a salary of > $50,000 or not based on demographic features. The model is trained on UCI Census Income Dataset.

## Intended Use

This model has been trained as part of a class project of Udacity's MLDevOps Nanodegree.

## Training Data

The training data has the following specifications:
- 14 features + label
- 8 categorical and 6 numerical features
- 32,561 entries
- 80/20 train-test split

## Evaluation Data

For evaluation, the test set is used, which comprises 20% of the total data.

## Metrics

Please see below for the metrics used and the respective performance:

-----------Overall model performance-----------
Precision: 0.7446471054718478
Recall: 0.5980891719745223
F-Beta: 0.6633698339809255

## Ethical Considerations

There seem to be significant differences in the predictions when looking at the sliced performance (e.g., for race or nationality). These should be considered when using the model, as it might lead to an increased number of FNs or FPs, dependent on circumstances.

## Caveats and Recommendations

Models other than Random Forest Classifier could be tested. I did not perform any hyperparameter-tuning on the model, which might improve performance.
