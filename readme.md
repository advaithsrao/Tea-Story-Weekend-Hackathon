## The repository holds my submission to MachineHack Tea-Story weekend hackathon(https://machinehack.com/hackathons/teastory_weekend_hackathon_edition_2_the_last_hacker_standing/)


- The task at hand is to predict average weekly tea prices across major cities
- My approaches included 
	- Time series analysis of the average prices across the 7 cities as a function of datetime feature.
	        - Prophet by Facebook
	        - ARIMA
	- Multiple Regresion techniques by the likes of 
		- Lasso Regression
		- KNN Regressor
		- Support Vector Regressor

- To view the problem as a regression problem, one could
 	- Impute null values across individual prices
 	- Convert datetime inputs to ordinal(Gregorian)
 	- Fit a regressor model to interpolate price prediction for each target city.
 	- Average these prices to get the ultimate target prediction
 	- To test error function, one could use the errors across known prices across test set, that could be weighted to get errors(for eg: if the price for city_A through city_F is known, we could check for the RMSE across these prices and divide this value by 6)

- To view the problem as a Time series problem, one could use prophet by Facebook or ARIMA, 
	- Impute null values across average prices
	- Use Moving averages, Exponential Smoothing to smooth target values over time periods

- I found the best results with a LassoRegressor, that helped me get an RMSE score of 2.16454 on the problem.

## How to run 

1. Create a virtual envirnoment to install packages in
   ```python3 -m venv env```

2. Source into the env and install packages from the requirements.txt file
   ```
   source env/bin/activate
   pip3 install -r requirements.txt
   ```

3. Run **regressor.ipynb** notebook
