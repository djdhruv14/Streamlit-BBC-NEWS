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
classifiers like Naive Bayes and Logistic Regression. We also confirmed that the use of a lower dimensional feature space made it easier to minimize noise introduced by infrequently used
terms.
 Class distribution:
- Business: 510
- Entertainment: 386
- Politics: 417
- Sport: 348
- Tech: 234
The imbalance justified stratified cross-validation to ensure all classes were represented correctly
in the training and validation sets. Stability of performance over the folds also confirmed the
suitability of this strategy

The models showed relatively close performance, but Naive Bayes slightly edged out its
competitors. This model's assumption of word independence and its probabilistic nature allowed
it to effectively classify text represented in sparse TF-IDF vectors. SVC and Logistic Regression
followed closely and were robust across all categories.
Random Forest and XGBoost, while typically strong for structured data, showed lower
performance in this context, possibly due to the challenges of handling extremely sparse matrices
and the lack of interaction terms inherent in bag-of-words models.


5.3 Category-Level Performance

Random Forest and XGBoost, otherwise very strong with regular data, were weaker here, in all
likelihood, because of challenges of dealing with super sparse matrices along with lack of
Sport and Business classes were most regularly categorized, which was likely due to the fact
that they were better represented in the dataset.

- Tech, as the smallest category, continued to perform strongly, which showed that the
distinctiveness of the vocabulary of articles covering tech made it easier for the model to
differentiate these well.

- There was a minimal overlap between Politics and Entertainment in certain misclassifications,
which could be due to thematic intersection between political opinion and cultural journalism.
Confusion matrices also ensured that the majority of errors were random and not systematically
biased toward particular labels.


5.4 Streamlit Interface Testing

A user interface was created utilizing Streamlit that supported real-time user-provided news
content classification. A test news article covering a Champions League match was correctly
identified as "Sport." The classifier handled domain-specific jargon (e.g., "Arsenal," "Champions
League," "possession stats") and was able to effectively use the trained model's internal 
implicit interaction terms in bag-of-words representations

Although the app yielded a runtime error in one case with regard to visualization rendering, this
was unrelated to the classification model but was subsequently resolved. Overall, the application
was a helpful demonstration of how the classifier might be applied in an
interactive environment for editors or reporters.


The results of this work are insightful into vanilla machine learning models' performance on
high-dimensional text-based classification problems. More importantly, the high performance of
Naive Bayes even more than high-end ensemble models like XGBoost tells much about simpler
algorithm use in certain scenarios. These findings corroborate earlier work that shows Naive
Bayes performing exceptionally well where features are sparse, abundant, and independent,
which is the case for TF-IDF representation.
One of the most important lessons from this project was how preprocessing and validation steps
can affect model performance. Preliminary results were misleadingly encouraging due to leakage
of data, where TF-IDF statistics had been computed for the whole data before division. This
violated independence assumption between training and test data and gave overly optimistic
scores in validation. Usage of a pipeline avoided this issue and produced a much more stable and
reproducible outcome.

Vectorization strategy also became a matter of choice. Limiting vocabulary to 3000 most
significant features prevented overfitting, and added better interpretability and efficiency in
training as well. This is especially crucial in the case of linear models that are prone to being
affected by unnecessary or rare features. Additionally, stratified cross-validation also resolved
class imbalance, providing more reliable model evaluation.

While the excellent performance, there are still some areas for improvement. While precision and
recall were always high, the lack of in-depth misclassification analysis limited our insight into
model failure. Future work should include a careful analysis of confusion matrices and example level errors to inform feature improvements.

Furthermore, this project did not play with advanced NLP methods such as word embeddings
(Word2Vec, GloVe) or transformer architectures such as BERT, which would potentially have
had gains, especially for nuanced or context-rich text. The trend of mashing up velocity of Naive
Bayes with richness of neural architecture would be of interest.

From an application standpoint, the use of this model in a web interface offered an added layer of
real-world testing. The ability to predict labels in real time attests to the model's usability and
strength. With some engineering improvements, this system would be the foundation of a
recommendation engine or an automatic news tagging system
