
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
from datetime import datetime
from statistics import median, mean, stdev

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #list of all stocks
ETFlist = ["BIL","BND","DBC","EEM","EFA","EMB","GLD","HYG","IJR","LQD","PFF","QQQ","SPY","TIP","TLT","VGK","VNQ","VPL","VT","VTI"]
    #Setting upper bound of "earliest month" - - will become lower bound
earliestmonth = "05/2022"
    #setting lower bound of "latest month" - - will become upper bound
latestmonth = "01/1980"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data/Pretreat Data/Feature Engineering Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for stock in ETFlist: #for each stock
    in_filename = "data\\"+stock + ".txt" #set datafile of stock as file to use
    #Load data
    #def load_doc(filename):
        #Column 0: Date ; Column 1: Opening Price ; Column 4: Closing Price
    data = pd.read_csv(in_filename, header=None, sep = ",") 
    
###CALCULATING DAILY GRANULARITY DATA###
    monthlist = [] #empty list to store month index
    dailyreturnlist = [] #empty list to store daily return (to be calculated)
    dates = [] #empty list to store dates - - comprised of month and year
    for i in range(len(data[0])): #for each day in the data
        if i == 0: #on the first day
            monthcount = 0 #set the month index counter to 0
            startmonth = data[0][i][0:2] #set starting month (derived from date value)
            currentmonth = startmonth #make start month the current month
            startyear = data[0][i][6:10] #set starting year (derived from date value)
            currentyear = startyear #make start year the current year
        if currentmonth != data[0][i][0:2]: #if the "current month" does not equal the actual current month
            monthcount += 1 #increase month index by 1
            currentmonth = data[0][i][0:2] #update current month
        if monthcount == 13: #by the 14th month will have enough data for first month's features
            firstmonth = data[0][i][0:2] #set this month as the first month
            firstyear = data[0][i][6:10] #set this year as the first year
        openprice = data[1][i] #take the opening price of this day
        closeprice = data[4][i] #take the closing price of this day
        dailyreturn = ((closeprice - openprice)/openprice)*100 #calculate the intra day return from open/close prices
        monthlist.append(monthcount) #append month index to list
        dailyreturnlist.append(dailyreturn) #append calculated return to list
        dates.append(data[0][i][0:3] + data[0][i][6:]) #append month/year date to list
    data["month"] = monthlist  #make this list a new column of dataframe
    data["dailyreturn"] = dailyreturnlist #make this list a new column of dataframe
    data["date"] = dates #make this list a new column of dataframe

###CALCULATING MONTHLY GRANULARITY DATA###      
    monthreturns = [] #create empty list for calculated monthly returns
    datedata = [] #create empty list for month/year date values 
    for i in range(monthcount + 1): #for each month in the month count 
        currentmonthdata = [j for j in data[4][data['month'] == i]] #find all closing prices for days in the current month
        endprice = currentmonthdata[-1] #set the month closing price as the last day's closing price
        if i > 0: #if not the first month in the list
            pastmonthdata = [j for j in data[4][data['month'] == i-1]] #collect the closing prices of the past month   
            startprice = pastmonthdata[-1] #set the starting price of the month as the closing price of the last day of the prior month
        else: #if the first month
            startprice = [j for j in data[1][data['month'] == i]][0] #set the starting price as the opening price for the first day of the month
        mreturn = ((endprice - startprice)/startprice)*100 #calculate monthly return from ending and starting prices
        monthreturns.append(mreturn) #append return to monthly returns list
        date = data["date"][data['month'] == i].iloc[0] #calculate month date based on date value of first row in the list (they should all be the same, but due to differences in length, the first row is guaranteed)
        datedata.append(date) #append this month/year date value to list
        
    returnlabels = ["month","date","nextreturn"] #set first three column labels for return data
    for i in range(12): #for 12 months
        returnlabels.append(f"tminus{2+i}") #add column label for each monthly return
    for i in range(20): #for 20 days
        returnlabels.append(f"day{i+1}") #add column label for each daily return
    globals()[f"{stock}_returndata"] = pd.DataFrame(columns=returnlabels) #use these labels to make empty dataframe with unique name based on stock value
    globals()[f"{stock}_tplus1returns"] = pd.DataFrame(columns=['date'])

###GETTING CUMULATIVE MONTHLY RETURN FEATURES###   
    for i in range(13, monthcount): #for all feasible months (14th month to second to last) - - only months with complete features
        if datetime.strptime(datedata[i],"%M/%Y") < datetime.strptime(earliestmonth,"%M/%Y"): #determine if current month is less than the current earliest month value
            earliestmonth = datedata[i] #update earliest month value -- should get lower
        if datetime.strptime(datedata[i],"%M/%Y") > datetime.strptime(latestmonth,"%M/%Y"): #determine if current month is greater than the current latest month value
            latestmonth = datedata[i] #update latest month value -- should get higher
        monthdata = monthreturns[i - 13:i - 1] #find the 12 monthly returns from t - 2 to t - 13 for current month
        datesspec = datedata[i] #set the specific date as the current date
        monthdata = [sum(monthdata[0:x:1]) for x in range(0, len(monthdata)+1)][1:] #create list of cumulative monthly returns - used for monthly return features
        nextreturn = monthreturns[i+1] #set the following months return - will be used to engineer label
        
###GETTING CUMULATIVE DAILY RETURN FEATURES###      
        daydata = [j for j in data["dailyreturn"][data['month'] == i]] #create list of all day returns in the current month
        if len(daydata) < 20: #if fewer than 20 returns
            need = 20 - len(daydata) #calculate the disparity
            added = [0]*need #make list of zeros based on disparity
            daydata = daydata + added #add list of zeros to end of daily returns
        elif len(daydata) > 20: #if more than 20 returns in this month
            daydata = daydata[:20] #take the first 20
        daydata = [sum(daydata[0:x:1]) for x in range(0, len(daydata)+1)][1:] #make list of 20 returns into cumulative values

###CREATING DATAFRAME WITH CUMULATIVE FEATURES AND LABEL###
        alldata = [i] + [datesspec] + [nextreturn] + monthdata + daydata #create row of data consisting of index, date label, next return, cumulative monthly returns, and cumulative daily returns
        globals()[f"{stock}_returndata"].loc[len(globals()[f"{stock}_returndata"])] = alldata #add this row of data to the stock's returns dataframe
        globals()[f"{stock}_tplus1returns"].loc[datesspec] = nextreturn

###FINDING NORMALIZATION DATA###        
currentdate = earliestmonth #set current date (start) as earliest month - - calculated earlier
mean_table = pd.DataFrame(columns=[returnlabels[1]] + returnlabels[3:]) #create empty dataframe of cross sectional means per date per feature
sd_table = pd.DataFrame(columns=[returnlabels[1]] + returnlabels[3:]) #create empty dataframe of cross sectional standard deviations per date per feature
median_table = pd.DataFrame(columns=["date", "median"]) #create empty data frame of cross sectional median per date
count = 0
while datetime.strptime(currentdate,"%M/%Y") <= datetime.strptime(latestmonth,"%M/%Y"): #while current month is less than or equal to latest month (iterate through all feasible months)
    currentmonth = int(currentdate[0:2]) #make current month integer
    currentyear = int(currentdate[3:]) #make current year integer
    temp_date_table = pd.DataFrame(columns=returnlabels) #create empty dataframe to be used to get unaggregated data for each month - - will be used to aggregate
    for stock in ETFlist: #for each stock in list
        try: #attempt to add data to temporary list if this stock has data for the current month
            temp_date_table.loc[len(temp_date_table)] = globals()[f"{stock}_returndata"][:][globals()[f"{stock}_returndata"]['date'] == currentdate].values[0]
        except: #otherwise move to next stock
            continue
    median_table.loc[len(median_table)] = [currentdate, median(temp_date_table["nextreturn"])] #set the median of the following months stock returns as the month's median
    print(count)
    print(temp_date_table["nextreturn"])
    count += 1
    temp_mean_row = [] #create empty list to temporarily hold row values to calculate mean
    temp_sd_row = [] #create empty list to temporarily hold row values to calculate standard deviation
    for i in range(3, len(returnlabels)): #for each of the monthly and daily return features
        temp_mean = mean(temp_date_table[returnlabels[i]]) #calculate the mean returns for this feature
        temp_mean_row.append(temp_mean) #add mean to list
        if len(temp_date_table[returnlabels[i]]) == 1: #if only one stock for this month
            temp_sd = 1 #cannot calculate standard deviation with one value, set as 1 (otherwise divide by 0 issue)
        else: #if more than one stock this month
            temp_sd = stdev(temp_date_table[returnlabels[i]].values,temp_mean) #calculate standard deviation for feature using returns and mean
        temp_sd_row.append(temp_sd) #add sd to list
    mean_table.loc[len(mean_table)] = [currentdate] + temp_mean_row #append means for this month's features to mean dataframe
    sd_table.loc[len(sd_table)] = [currentdate] + temp_sd_row #append standard deviations for this month's features to standard deviation dataframe
    if currentmonth + 1 == 13: #if increasing month counter leads to 13 then year is over
        currentmonth = "01" #reset current month to january
        currentyear += 1 #increase year by 1
    else: #if not end of year
        currentmonth += 1 #increase current month counter by 1
        if currentmonth < 10: #if a single digit value
            currentmonth = "0" + str(currentmonth) #make a string with a 0 at start
        else: #if double digits
            currentmonth = str(currentmonth) #make as string
    currentdate = currentmonth + "/" + str(currentyear) #combine current month and current year to update current date

###NORMALIZING FEATURES###    
for stock in ETFlist: #for each stock in list
    globals()[f"{stock}_finalfeatures"] = pd.DataFrame(columns=returnlabels[1:]) #create an empty dataframe to house normalized features (as well as date value and classifier label)
    for date in globals()[f"{stock}_returndata"]["date"].values: #for each date in this stocks return history
        temp_norm_row = [date] #create an empty list to fill with normalized features and labels
            #if the next month's stock return is greater than the cross sectional median
        if globals()[f"{stock}_returndata"]["nextreturn"][globals()[f"{stock}_returndata"]['date'] == date].values > median_table["median"][median_table["date"] == date].values:
            temp_norm_row.append(1) #set label as a 1 - buy stock
        else: #if the next month's stock return is not greater than the cross sectional median
            temp_norm_row.append(0) #set label as a 0 - do not buy stock/sell stock
        for i in returnlabels[3:]: #for each of the features
            temp_x = globals()[f"{stock}_returndata"][i][globals()[f"{stock}_returndata"]['date'] == date].values #find the raw return value given the current month
            temp_mean = mean_table[i][mean_table['date'] == date].values #find the cross sectional mean given the current month
            temp_sd = sd_table[i][sd_table['date'] == date].values #find the cross sectional standard deviation given the current month
            temp_z = (temp_x - temp_mean) / temp_sd #calculate z-score to normalize return
            temp_norm_row.append(temp_z[0]) #add z-score to normalized features list
        globals()[f"{stock}_finalfeatures"].loc[len(globals()[f"{stock}_finalfeatures"])] = temp_norm_row #append this list as the next row in the stock's normalized features values
    #sending each finalized dataframe to a csv        
    globals()[f"{stock}_finalfeatures"].to_csv(f"{stock}_normalized_features.csv",index=False)
    globals()[f"{stock}_tplus1returns"].to_csv(f"{stock}_tplus1returns",index=True)     
