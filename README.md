In this report, supervised machine learning models are applied to the issue of automatic
document classification, in our case on news stories. Based on a publicly available BBC news
dataset consisting of 2225 news stories, five in count (business, entertainment, politics, sport, and
tech), the project endeavored to develop a robust classifier that would label unseen text
accurately. Early stages of the project were beset by serious data leakage and model overfitting
issues, resulting in misleadingly high cross-validation scores and bad generalization. These
issues were addressed through the use of pipeline-based preprocessing, correct cross-validation
practices, and hyperparameter tuning. Final models exhibited exemplary performance, with
Naive Bayes achieving a cross-validation accuracy of 97.3%, followed closely by Support Vector
Machines and Logistic Regression. Random Forest and XGBoost, although widely used,
performed poorly in this particular high-dimensional sparse feature environment. The results
reaffirm the value of methodological discipline and emphasize that simpler, well-tuned models
are capable of beating more complicated algorithms for natural language processing (NLP) tasks

METHODOLOGY

Dataset Description

The dataset employed in this study is a publicly available dataset of 2225 BBC news
stories, each of which is labeled with one of five categories: business, entertainment,
politics, sport, or tech. The categories vary in size, with business and sport the largest,
and tech the smallest. The data were divided into a training set of 1490 samples and a test
set of 735 samples.

Data Preprocessing

Preprocessing during the first stage included basic text cleaning (lowercasing,
punctuation removal), tokenization, and vectorization. Initial technique utilized TF-IDF
Vectorizer to the entire dataset before cross-validation. This led to data leakage since the
model gained knowledge of the vocabulary and term frequencies from the entire dataset,
including validation data.
To mitigate this, we imposed a robust preprocessing pipeline using scikit-learn's Pipeline
class. This encapsulated the vectorization and classification around so that each fold in
the cross-validated procedure learned preprocessing statistics independently. We also
limited the vectorizer to 3000 features to avoid overfitting to noisy or rare terms.

Cross-Validation Strategy

Instead of simple K-Fold cross-validation, we employed Stratified K-Fold to maintain
proportional class distribution across folds. This was required since the dataset was classimbalanced (for example, tech had significantly fewer examples than business)
Model Selection and Training
We experimented with five machine learning models:
• Multinomial Naive Bayes: Best suited for discrete feature distributions like term
frequencies.
• Support Vector Classifier (SVC): Employed a linear kernel with regularization
parameter C=0.1.
• Logistic Regression: Regularized during training with C=0.1.
• Random Forest: 100 tree ensembles with max_depth=10.
• XGBoost: Gradient boosting with optimized learning rate and depth.
All the models were added to the pipeline and then evaluated on 5-fold stratified crossvalidation.
Evaluation Metrics
The overall metric was the classification accuracy. We also computed class-wise
precision, recall, and F1 scores to know model performance with respect to classes

Dataset Summary and Cleaning Impact

After TF-IDF transformation, the data consisted of a 3000-dimensional sparse matrix.
Vocabulary size constraint was crucial in preventing overfitting as initial models that had been
trained on unlimited vocabulary were producing excessively large scores. It also facilitated the
achievement of faster training time and more interpretable models when employing simpler
classifiers like Naive Bayes and Logistic Regression. We also confirmed that the use of a lowerdimensional feature space made it easier to minimize noise introduced by infrequently used
terms.
 Class distribution:
- Business: 510
- Entertainment: 386
- Politics: 417
- Sport: 348
- Tech: 234
The imbalance justified stratified cross-validation to ensure all classes were represented correctly
in the training and validation sets. Stability of performance over the folds also confirmed the
suitability of this strategY

