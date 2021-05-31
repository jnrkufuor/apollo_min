#import packages
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import os
from apollo.utils.util import Util
from GoogleNews import GoogleNews
from goose3 import Goose
from goose3.configuration import Configuration
from datetime import datetime
ut = Util()

class Scraper:

    def __init__(self, company_list, period="None", date_range="None"):
        ''' Initialization function for data scraper 

            :param period: Optional choose period (period and custom day range should not set together)
            :param date_range: A list of the start and end date you want to fetch news on.  (mm/dd/yyyy) [Start, End] ~ ["02-12-2002","02-12-2020"] 
            :param company_list: A list of the companies you want to filter the news on. Example: ["MSFT","APPL","VX"]
        '''
        if period != "None":
            self.period_exists=True
            self.period = period
            self.gns = GoogleNews(period='7d')
        elif date_range != "None":
            self.date_range_exists=True
            self.date_range = date_range
            self.gns = GoogleNews(start=date_range[0], end=date_range[1])
        else:
            self.date_range_exists=False
            self.periodexists=False
            self.period=period
            self.date_range=date_range
            self.gns = GoogleNews()
        self.company_list = company_list

    def set_date_range(self, date_range):
        ''' Function to set the date range for the scraper object

            :param date_range: A list of the start and end date you want to fetch news on.  (mm/dd/yyyy) [Start, End] ~ ["02-12-2002","02-12-2020"] 
        '''
        self.gns.clear()
        self.period="None"
        self.date_range = date_range
        self.gns.set_time_range(start=date_range[0], end=date_range[1])

    def set_period(self, period):
        ''' Function to set the period

            :param period: Optional choose period (period and custom day range should not set together)
        '''
        self.gns.clear()
        self.date_range="None"
        self.period = period
        self.gns.set_period(period)

    def set_company_list(self, company_list):
        ''' Function to set the list of companies to filter on

            :param company_list: A list of the companies you want to filter the news on. Example: ["MSFT","APPL","VX"]
        '''
        self.company_list = company_list

    def fetch_news_data(self,num_of_articles):
        ''' Function to fetch news based on set parameters

            :param num_of_articles: number of articles to fetch
            :return newsframe: dataframe with news content
        '''
        newsframe = pd.DataFrame()
        for company in self.company_list:
            self.gns.search(company)
            df_comp = pd.DataFrame(self.gns.result()).iloc[0:10, ]
            newsframe = newsframe.append(df_comp)
            self.gns.clear()
        
        newsframe["content"]=np.zeros(len(newsframe.iloc[:,1]))
        g = Goose()
        with Goose({'http_timeout': 5.0}) as g:
            pass
        for i in range(0,num_of_articles):
            try:
                newsframe.iloc[i,7]=(g.extract(url=newsframe.iloc[i,5])).cleaned_text
            except:
                newsframe.iloc[i,7]="Missing_Article"
        return newsframe

    def check_status(self):
        ''' Function to print instance variables
        '''
        print(self.company_list)
        print(self.gns)
        if self.period != "None":
            print(self.period)
        if self.date_range != "None":
            print(self.date_range)
    
    def fetch_financial_data(self):
        ''' Function to print instance variables
        '''
        #download price timeseries from Yahoo Finance
        price_corr = []
        volume_corr=[]
        if (self.date_range == "None"):
            print("Date range has not been set, set using set_date_range() function")
            return [price_corr,volume_corr]
        symbols = sorted(self.company_list)
        symbols_data ={}
        print("Downloading {} files".format(len(symbols)))
        for i, symbol in enumerate(symbols):
            try:
                df = web.DataReader(symbol,'yahoo', self.date_range[0],  self.date_range[1])
                df = df[['Adj Close','Volume']]
                symbols_data[symbol]=df
            except KeyError:
                print("Error for {}".format(symbol))
                pass
        
        #pre-process timeseries data
        print("Preprocessing Data...")
        index = pd.date_range(start=self.date_range[0], end=self.date_range[1], freq='D')     # initialize an empty DateTime Index
        df_price = pd.DataFrame(index=index, columns=symbols)               # initialize empty dataframes
        df_volume = pd.DataFrame(index=index, columns=symbols)
        
        print("Aggregate all symbols into a price dataframe and volume dataframe...")
        # Aggregate all symbols into a price dataframe and volume dataframe
        for symbol in symbols:
            #symbol_df = symbols_data[symbol].set_index('Date')
            symbol_df = symbols_data[symbol]
            #symbol_df = pd.read_csv(os.path.join(data_dir, symbol+".csv")).set_index('Date')
            symbol_df.index = pd.to_datetime(symbol_df.index)

            df_price[symbol] = symbol_df['Adj Close']
            df_volume[symbol] = symbol_df['Volume']
        
        # Let's drop the dates where all the stocks are NaNs, ie., weekends/holidays where no trading occured
        #df_price = df_price.bfill(axis='rows')   
        #df_price = df_price.ffill(axis='rows')
        df_price.dropna(how='all', inplace=True)
        df_volume.dropna(how='all', inplace=True)
        assert((df_price.index == df_volume.index).all())
        pd.isnull(df_price).sum()
        
        #Obtain Percentage Change and Correlation
        #We need to convert prices to percent change in price as opposed to the actual $ price. This is because stocks with very similar prices can behave very differently and vice-versa. For e.g., 
        # if a stock moves from  100 to 110, we want the price column to say 10% (indicating the change).

        #However, for volume, we will retain magnitude.
        df_price_pct = df_price.pct_change().dropna(how='all')
        
        #calculate correlations
        print("Creating correlation matrices...")
        price_corr = df_price_pct.corr()
        volume_corr = df_volume.corr()
        print("Done")
        return [price_corr,volume_corr]
        
        


if __name__ == "__main__":
    s = Scraper(["AAPL", "MSFT","MMM","AXP","AMGN","CAT","GS","HD","TRV","JPM","WMT","CVX","INTC"])
    s.set_period("7d")
    s.set_date_range(["02-01-2015","02-12-2015"])

    #s.check_status()
    #news = s.fetch_news_data(10)
    #Util.print_to_csv(news,"news.csv","data")
    [price,vol] = s.fetch_financial_data()
    #ut.print_to_csv(news,"news.csv")
    ut.print_to_csv(price,"price.csv")
    ut.print_to_csv(vol,"vol.csv")