# INR - USD exchange rate forecasting using Bayesian VAR 

In this project, we evaluate the performance of bayesian Vector Auto Regressive models in the context of Indian exchange rate forecasting. Various priors are compared.  

Presented this work as a talk titled, "Bayesian forecasting of Rupee/Dollar exchange rate: Does Minnesota prior matter?" at the International Conference On Financial Markets And Corporate Finance held in IIT Kharagpur, India.

A detailed explanation of the approach and discussion of the results can be found in the [report.](Report.pdf)

I have also explored the time aware deep learning models like LSTMs for exchange rate forecasting. The project can be found  [in this repository.](https://github.com/rakshita95/DeepLearning-time-series)

## Files and their Functions

* Report: Writeup of the details of the project
* data.csv: Contains data used in forecasting. Obtained from RBI's website
* bayes_var: Main matlab file. Compares and evaluates the various models
* mlag2.m: Helper file

