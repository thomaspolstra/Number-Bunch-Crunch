# Team 35: Volatility Estimation

by Waleed Ahmed, Palak Arora, Cameron Cinel, George Mitchell, and Thomas Polstra

This repository contains code for training three different models for predicting stock volatility given historical data.
The three models are:
- [GARCH](/GARCH) (General Autoregressive Conditional Heteroskedasticity)
- [Random Forest](/Random-Forest)
- [LSTM](/LSTM) (Long Short Term Memory)

In this README, we explain the data gathering process, the preprocessing, and the models


## Table of Contents
- [Data Gathering](#Data Gathering)
- [Preprocessing](#Preprocessing)
- [Models](#Models)
  - [GARCH](#GARCH)
  - [Random Forest](#Random Forest)
  - [LSTM](#LSTM)

## Data Gathering

Data was gathered using the Python package [`yfinance`](https://pypi.org/project/yfinance/).
Data was collected for across 4 different sectors (tech, biotech, healthcare, and industrial) with 5 stocks in each
sector.
Historical prices were collected from 1-1-2013 to 12-31-2022.

## Preprocessing

The data from `yfinance` includes a significant amount of information, including closing prices, trading volume, and
high and low prices.
As the stocks were chosen such that they were traded for the entirety of our chose time period, no data cleaning was
needed.

For our purposes, just the adjusted close price was used for each of the 20 stocks.
Additionally, as returns (ratio of current to previous price) are more relevant to volatility, we looked at the log
returns of the prices.
The log returns were chosen as they are simply a difference instead of a ratio.
For some of our models, additional data processing was used, which is explained in their relevant sections.

We also preformed some EDA in order to verify that our time series were stationary, in order to satisfy the assumptions
of the GARCH model.

## Models

### GARCH
A GARCH($`p`$, $`q`$) model is an autoregressive model for time series data that approximates the variance of the series
as a function of the previous time periods' error terms.
Specifically, the variance is modeled as a linear series of the previous $`q`$ returns and the previous $`p`$ variances.

The best $`p`$ and $`q`$ values were chosen via cross-validation, with most stocks performing best with $`p=q=1`$.
We used the first 8 years of returns as our training data and our final 2 years as the testing data.
Due to the architecture of the GARCH model, a validation set is not required.

### Random Forest
We trained a random forest regression to take in a rolling window of observed volatilities and predict the next day's
expected observed volatility.
Random forest regression works by training multiple decision trees to make predictions on the stock's volatility.
Then, the average prediction amongst all trees is chosen as the output of the forest in order to avoid overfitting to
the dataset.

### LSTM
An LSTM is a type of recurrent neural network (a network that connects back to itself) that is designed in order to deal
with long input sequences.
The key feature of LSTMs is their use of memory cells, which allow networks to learn when to remember, forget, and
recall stored memory, allowing them to avoid the vanishing gradient problems of other recurrent networks.

We trained our model using the first 8 years of data, validated it on the next year, and tested it on the final year.
Through experimentation, we found the best architecture involved a 4 layer LSTM with 128 neurons each followed by 3
fully connected layers.
The LSTM was trained to take in a rolling window of $`k`$ days worth of observed volatility, and output the expected
observed volatility for the next day.
In our experiments, 30 was found to be the best rolling window size.

To train your own LSTM in this repo, choose a collection of `n` stocks, a start date `d1-m1-y1` and end date `d2-m2-y2`.
Download the data through `yfinance` and save it as a pickle file with `pandas` in the `/data` directory with the name
format `tickers_data_n_d1_m1_y1_d2_m2_y2.pkl`.
Then in `/LSTM/configs`, create your own `.json` in the same format and make sure to enter `n` as the parameter for
`n_tickers`.
Finally, the LSTM can be trained and tested via running
```commandline
python main.py your_config_name_here.json
```
