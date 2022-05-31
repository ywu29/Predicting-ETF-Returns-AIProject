
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
from datetime import datetime
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #list of all stocks
ETFlist = ["BIL","BND","DBC","EEM","EFA","EMB","GLD","HYG","IJR","LQD","PFF","QQQ","SPY","TIP","TLT","VGK","VNQ","VPL","VT","VTI"]
    #Number of years to predict over
years_predict = [1, 5, 10]
    #Number of stocks to buy/sell
topN = [3, 4, 5]
    #starting balance
start_bal = 250000        
    #purchase/sale charge
charge = 8.95
    #current stocks in portfolio
end_date = '02/2022'
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Making Predictions Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ###GETTING NECESSARY DATA AND CREATING LABELS###
#STOCK DATA AND MODELS
for stock in ETFlist: #for each stock
    globals()[f"{stock}_normalized_features"] = pd.read_csv(f"{stock}_normalized_features.csv") #import normalized data
    globals()[f"{stock}_model"] = load_model(f'model.{stock}') #import model
    globals()[f"{stock}_tplus1returns"] = pd.read_csv(f"{stock}_tplus1returns",index_col = 0) #import returns data for portfolio

#FEATURE LABELS
returnlabels = [] #set empty list
for i in range(12): #for 12 months
    returnlabels.append(f"tminus{2+i}") #add column label for each monthly return
for i in range(20): #for 20 days
    returnlabels.append(f"day{i+1}") #add column label for each daily return
    
PortfolioKPI = pd.DataFrame(columns=['Number of Years', 'Number of EFTs', 'Largest Drawdown', 'Total Return', 'Drawdown to SPY', 'Return to SPY'])

for stock_nums in topN: #for each portfolio split
    for years_num in years_predict: #for each portfolio run time
        #DATE LABELS    
        end_month = end_date[:2] #set ending month for prediction data
        end_year = int(end_date[3:]) #set ending year for prediction data
        start_month = end_month #based on year data (and prediction tendency to lose first observation, starting month will be ending month)
        start_year = str(end_year - years_num) #starting year will always be end year minus the the number of years to predict
        current_month = start_month #set current month
        current_year = start_year #set current year
        current_date = start_month + "/" + start_year #set current date
        start_date = current_date #current date is start date
        datelabels = [] #empty dataframe - - these will be used as headers
            
        while datetime.strptime(current_date,"%M/%Y") < datetime.strptime(end_date,"%M/%Y"): #while current date less than end date
            if int(current_month) + 1 == 13: #if at twelfth month
                current_month = '01' #set current month to january
                current_year = str(int(current_year) + 1) #increase year by 1
            else: #if not last month of year
                current_month = ('0' + str(int(current_month) + 1))[-2:] #increase current month by one and add a zero for string if single digit
            current_date = current_month + "/" + current_year #update current date
            datelabels.append(current_date) #add current date to labels
        predictions = pd.DataFrame(columns = datelabels) #make predictions dataframe with date labels 
               
            ###MAKING PREDICTIONS###
        for stock in ETFlist: #for each stock
            features = globals()[f"{stock}_normalized_features"][returnlabels].to_numpy() #dataframe of features
            label = globals()[f"{stock}_normalized_features"]['nextreturn'].to_numpy() #array of labels
                #find index of starting date
            start_index = globals()[f"{stock}_normalized_features"][globals()[f"{stock}_normalized_features"]['date'] == start_date].index.values[0]
                #find index of ending date
            end_index = globals()[f"{stock}_normalized_features"][globals()[f"{stock}_normalized_features"]['date'] == end_date].index.values[0]
            
            TSG = TimeseriesGenerator( #create time series generator with data specifying to only use between start and end dates
                features,
                label,
                1,
                sampling_rate=1,
                stride=1,
                start_index=start_index,
                end_index=end_index,
                shuffle=False,
                reverse=False,
                batch_size = 1
            )
            pred = globals()[f"{stock}_model"].predict(TSG, verbose=0) #make predictions with the time series and the ETF's model
            predictions.loc[stock] = [i[0] for i in pred] #make the predictions a row of dataframe with ETF name as index
        
        ###BACKTESTING PORTFOLIO PERFORMANCE###    
            #create performance table for KPIs
        performance = pd.DataFrame(columns=['Stocks Held', 'Monthly Return', 'Total Return', 'Percent Return', 'Percent Total Return', 'SPY Performance','SPY Comparison'])
            #whether or not in a drawdown
        portfolio = []
            #end date
        in_drawdown = False    
            #list of drawdowns
        drawdowns = []
            #Spy's starting balance
        start_bal_SPY = 250000
            #whether or not SPY in a drawdown
        in_drawdown_SPY = False  
            #list of drawdowns SPY
        drawdowns_SPY = []
        
        current_bal = start_bal #set current balance as starting balance
        max_balance = current_bal #setting max_balance as current balance
        min_balance = current_bal #setting min balance as current balance - - will be updated for each new drawdown
        percent_total_return = 0 #setting portfolio percent return as 0
        
        current_bal_SPY = start_bal_SPY #set current balance for SPY as starting balance for SPY
        max_balance_SPY = current_bal_SPY #setting max_balance for SPY as current balance for SPY
        min_balance_SPY = current_bal_SPY #setting min balance for SPY  as current balance - - will be updated for each new SPY drawdown
        percent_total_return_SPY = 0 #setting SPY percent return as 0
        
        perc_portfolio = 1/stock_nums #stocks will always be evenly distributed so percentage of portfolio is equal to 1/number of stocks
        for month in datelabels: #go month by month to choose best portfolio
            best_N = predictions[month].sort_values(ascending=False)[:stock_nums].index.values #take the top N portfolio names
            new_portfolio = [i for i in best_N] #add these portfolios to portfolio
            dollar_return = 0 #set dollar return at zero for new month
            percent_return = 0 #set percent return at zero for new month
            dollar_return_SPY = 0 #set dollar return for SPY at zero for new month
            percent_return_SPY = 0 #set percent return for SPY at zero for new month
            for stock in new_portfolio: #for each stock in new portfolio
                if stock not in portfolio: #if not currently in portfolio
                    current_bal = current_bal - charge #buy stock charge
            for stock in portfolio: #for each stock in old portfolio
                if stock not in new_portfolio: #if not in new portfolio
                    current_bal = current_bal - charge #sell stock charge
            portfolio = new_portfolio #set new portfolio as portfolio
            for stock in portfolio: #for each stock in portfolio
                percent_return = percent_return + globals()[f"{stock}_tplus1returns"].loc[month].values[0]/100 #CALCULATE THE PERCENT RETURN
                dollar_return = dollar_return + current_bal*perc_portfolio*percent_return #CALCULATE THE DOLLAR RETURN
            current_bal = current_bal + dollar_return #ADD TO TOTAL RETURN
            percent_total_return = percent_total_return + percent_return
            
            if current_bal > max_balance and in_drawdown == True: #if in a draw down and return passes max 
                drawdown = (min_balance - max_balance)/max_balance #calculate drawdown
                drawdowns.append(drawdown) #add drawdown to list of drawdowns
                max_balance = current_bal #reset max balance to current
                min_balance = current_bal #reset min balance to current
                in_drawdown = False #set drawdown to false since we've passed the max and are no longer in a drawdown
            elif current_bal < max_balance and in_drawdown == False: #if below max balance and not yet in drawdown
                in_drawdown = True #set into drawdown
            elif current_bal > max_balance: #if current balance greater than max
                max_balance = current_bal #update max balance to current
            if current_bal < min_balance: #if balance has dropped below minimum
                min_balance = current_bal #update minimum balance
                
            percent_return_SPY = globals()["SPY_tplus1returns"].loc[month].values[0]/100 #CALCULATE THE PERCENT RETURN FOR SPY
            percent_total_return_SPY = percent_total_return_SPY + percent_return_SPY  #CALCULATE THE TOTAL PERCENT RETURN FOR SPY
            dollar_return_SPY = dollar_return_SPY + current_bal_SPY*percent_return_SPY #CALCULATE THE DOLLAR RETURN FOR SPY
            current_bal_SPY = current_bal_SPY + dollar_return_SPY #ADD TO TOTAL RETURN
            if current_bal_SPY > max_balance_SPY and in_drawdown_SPY == True: #if SPY in a draw down and return passes max 
                drawdown_SPY = (min_balance_SPY - max_balance_SPY)/max_balance_SPY #calculate drawdown for SPY
                drawdowns_SPY.append(drawdown_SPY) #add drawdown to list of drawdowns for SPY
                max_balance_SPY = current_bal_SPY #reset max balance to current for SPY
                min_balance_SPY = current_bal_SPY #reset min balance to current for SPY
                in_drawdown_SPY = False #set drawdown to false since we've passed the max and are no longer in a drawdown for SPY
            elif current_bal_SPY < max_balance_SPY and in_drawdown_SPY == False: #if below peak and not in drawdown
                in_drawdown_SPY = True #set in drowdown for SPY to true
            elif current_bal_SPY > max_balance_SPY: #if current balance greater than max
                max_balance_SPY = current_bal_SPY #update max balance to current
            if current_bal_SPY < min_balance_SPY: #if balance has dropped
                min_balance_SPY = current_bal_SPY #update minimum balance for SPY   
                
            #ADD KPIS TO performance dataframe
            performance.loc[len(performance)] = [portfolio, dollar_return, current_bal, percent_return*100, percent_total_return*100, current_bal_SPY, percent_return_SPY]
        performance.to_csv(f"{stock_nums}S_{years_num}Y_performance",index=False)
        try: #try to find maximum drawdown
            max_drawdown = max(drawdowns)*100 #Calculate largest drawdown
        except: #if empty list
            if in_drawdown == True:
                max_drawdown = (min_balance - max_balance)*100/max_balance
            else:
                max_drawdown = 'No Drawdown' #no drawdown to record
        total_return = (current_bal - start_bal)*100/start_bal #Calculate total return
        try: #try to find maximum drawdown
            max_drawdown_SPY = max(drawdowns_SPY)*100 #Calculate SPY's largest drawdown
        except: #if empty list
            if in_drawdown_SPY == True:
                max_drawdown_SPY = (min_balance_SPY - max_balance_SPY)*100/max_balance_SPY
            else:
                max_drawdown_SPY = 'No Drawdown' #no drawdown to record
        try:
            drawdown_to_SPY = (max_drawdown - max_drawdown_SPY)*100/max_drawdown_SPY #calculate relative drawdown to SPY - - negative if not as bad of a drawdown
        except:
            drawdown_to_SPY = 'No Drawdown'
        total_return_SPY = (current_bal_SPY - start_bal_SPY)/start_bal_SPY #Calculate SPY's total return
        total_return_to_SPY = (total_return - total_return_SPY)/total_return_SPY #Calculate total return relative to SPY - - positive if out performs SPY
        PortfolioKPI.loc[len(PortfolioKPI)] = [years_num, stock_nums, max_drawdown, total_return, drawdown_to_SPY, total_return_to_SPY] #ADD KPIS to KPI dataframe
        
        #GRAPH PORTFOLIO PERFORMANCE COMPARED TO SPY PERFORMANCE  AND SAVE TO JPGs
        fig,ax = plt.subplots() #create plot
        plt.plot(list(range(years_num*12)), list(performance['Total Return']), 'green', label='Portfolio Earnings') #plot portfolio earnings line
        plt.plot(list(range(years_num*12)), list(performance['SPY Performance']), 'blue', label='SPY Earnings') #plot SPY earnings line
        plt.title(f'{stock_nums} ETFs | {years_num} Years: Earnings') #add title
        plt.legend() #add legend
        ax.spines['right'].set_visible(False) #hide right axis
        ax.spines['top'].set_visible(False) #hide top axis
        
        plt.figure() #plot figure
        plt.savefig(f"{stock_nums}_{years_num}_earnings.jpg")
        
        fig,ax = plt.subplots() #new figure
        plt.plot(range(years_num*12), list(performance['Monthly Return']), 'green', label='Portfolio Return') #plot portfolio return line
        plt.plot(range(years_num*12), list(performance['SPY Comparison']), 'blue', label='SPY Return') #plot SPY return line
        plt.title(f'{stock_nums} ETFs | {years_num} Years: Return') #add title
        plt.legend() #add legend
        ax.spines['right'].set_visible(False) #hide right axis
        ax.spines['top'].set_visible(False) #hide top axis
        
        plt.show() #show plot
        plt.savefig(f"{stock_nums}_{years_num}_return.jpg")
#SAVE KPIs to CSV
PortfolioKPI.to_csv("PortfolioKPI",index=False) 
