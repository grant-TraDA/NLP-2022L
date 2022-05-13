## Project Overview
The methodology for working on the solution involves incremental development of deliverables and summarizing progress during 3 milestones. This is because there are strong dependencies between the components of the project. The requirements for successive increments are as follows:
#### Milestone 1.:
1. Literature overview
2. Exploratory Data Analysis 

The results of the work performed have been summarized in the `docs\punctuation_restoration_milestone1.pdf` file

#### Milestone 2.:
1. Reiteration of the step of exploratory data analysis aimed at identifying symbols denormalized for the Polish language and identification of recordings for which transcription data is missing.
2. Data cleaning – i.e. the elimination of symbols denormalized for the Polish language.
3. Morphological analysis of the text with the use of Morpheus2 - the main goals were to verify the correctness of data cleaning in terms of the presence of denormalized symbols in the text (also re-checking the balance of classes), analysis of morphosyntactic markers.
4. Creating a rule-based baseline model - building a dictionary and writing rules writing regex rules that will apply a dictionary approach
5. Preparation of the first version of the implementation of the proprietary approach on top of existing solutions

The codes corresponding to the individual requirements have been placed in the following locations:
1.	`notebooks/eda.ipynb` file
2.	`notebooks/feature_engineering/data_cleaning.ipynb` file
3.	`notebooks/feature_engineering/morfeusz.ipynb` file
4.	`notebooks/baseline/baseline_model.ipynb` file (alongside supporting files for data preparation and cleaning)
5.	`src/` directory

#### Results:

We distinguished two approaches of evaluation of the prepared models. They differ in handling the blanks (no punctuation)	- whether we include them as a separate class during calculating the scores or not. We evaluated and compared the performance of three models:

* baseline with pauses - a baseline model including information about pauses after each word (from timestamps files)
* baseline without pauses - a baseline model based only on regular expressions (without information from timestamps files)
* neural model - Herbert model from one of the solutions of the competition

All the models were evaluated using `f1-weighted` score.

|                    | Baseline with pauses | Baseline without pauses | Neural model     |
|--------------------|----------------------|-------------------------|------------------|
|                    |                      |                         |                  |
| *Including blanks* | 0.8248               | 0.7895                  | 0.9289           |
|                    |                      |                         |                  |
| *Excluding blanks* | 0.3356               | 0.1261                  | 0.7229           |
|                    |                      |                         |                  |

We can observe that the difference in results between two considered approaches is significant for each of the models, especially in case of the baseline ones. It is caused by the fact that there is a large majority of blanks among the classes and therefore, this class has a great weight assigned. It is also a class recognized best by the models (due to its cardinality, mainly) what leads to the far better results obtained while including this class in the score calculation. The cardinality of the blank class is also the reason of small differences between the results of two baseline models while including blanks, and significantly higher difference while excluding them. In the second approach weight assigned to all the other punctuation marks increases greatly and therefore, resigning from the information about the pauses, which led to fullstops insertion (especially at the end of the records, so mostly a proper ones), causes a lower f1-weighted score.

|       | support | weights including blanks | weights excluding blanks |
|------:|--------:|-------------------------:|-------------------------:|
| blank | 33853.0 |                 0.845775 |                 0.000000 |
|     : |   322.0 |                 0.008045 |                 0.052163 |
|     ; |     0.0 |                 0.000000 |                 0.000000 |
|     , |  2498.0 |                 0.062409 |                 0.404665 |
|     . |  2568.0 |                 0.064158 |                 0.416005 |
|     - |   621.0 |                 0.015515 |                 0.100599 |
|     ? |   144.0 |                 0.003598 |                 0.023327 |
|     ! |    20.0 |                 0.000500 |                 0.003240 |

## Resources:
1. https://github.com/poleval/2021-punctuation-restoration
2. https://github.com/enelpol/poleval2021-task1
3. https://github.com/xashru/punctuation-restoration/tree/master/src
