# Na-ve-bayes-algorithm-to-classify-the-fetal-health

-Fetal Health classification Fetal_health.csv dataset contains 2126 records of features extracted from Cardiotocogram exams (Cardiotocogram is equipment used to read data about the fetus using ultrasound pulses, data like fetal heart rate (FHR), fetal movements, uterine contractions and more), fetal health was then classified by three expert obstetritians into 3 classes:

Normal 

Suspect 

Pathological 

The dataset consists of 22 columns as follow: 

Baseline value(Baseline Fetal Heart Rate)  accelerations(Number of acceleration per sec) 

fetal_movements(Number of Fetal movements per sec) 

uterine_contractions(Number of uterine contractions per sec) 

light_decelerations(Number of light decelerations per sec) 

severe_decelerations (Number of severe decelerations per sec) 

prolongued_decelerations( Number of prolongued decelerations per sec) 

abnormal_short_term_variability(Percentage of time with abnormal short term variability) 

mean_value_of_short_term_variability 

Percentage of time with abnormal long term variability 

Mean value of long term variability 

histogram_width(Width of the histogram made using all values from a record) 

histogram_min(Histogram minimum value) 

histogram_max(Histogram maximum value) 

histogram_number_of_peaks(Number of peaks in the exam histogram) 

histogram_number_of_zeroes(Number of zeros in the exam histogram) 

histogram_mode 

histogram_mean 

histogram_median 

histogram_variance 

histogram_tendency


fetal_health(values are 1 for Normal, 2 for suspect and 3 for pathological) It’s required to do the following:

1. Randomly select 30% of the dataset records and save them into fetal-healthTestData.csv file

2.Design and Implement a Naïve bayes algorithm to classify the fetal health using the 21 features above(i.e. baseline value, accelerations, fetal_movements … and so on) 

3. Use the extracted Test data records from Step 1 to test your implemented algorithm.
