## Math-exam-results

This project based on the Kagle competition. Link to the competition - [https://www.kaggle.com/competitions/gb-choose-tutors](https://www.kaggle.com/competitions/gb-tutors-expected-math-exam-results)

The main target of this repository is put into practice the knowledge about Decision Tree

Description:

In this competition your task will be to predict the mean math exam result (from 0 to 100 points) for students of tutors in test.csv. You will be given two datasets: train.csv (contains all features and the target) and test.csv (only features).


The evaluation metric is [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)

Competition Rules - You can only use these imports:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

Dataset Description

Id - unique identifier

age - tutor age

years_of_experience - tutor experience

lesson_price - price for 1 lesson

qualification

physics - teaches physics

chemistry - teaches chemistry

biology - teaches biology

english - teaches english

geography - teaches geography

history - teaches history

mean_exam_points


As the result: fitted model predicted exam results. Metric $R^2$ showed 0.605 on Baseline. On the test data (the result of the competition) 0.95284
