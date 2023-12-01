# Classic Lib
import numpy as np
import yfinance as yf
import datetime as dt

# Visualization
import matplotlib.pyplot as plt


############## BackTesting Strategies ################


class BackTestingStrat():
    def __init__(self,symbol,SMA_Short, SMA_Long,start,end):
        self.symbol=symbol
        self.SMA_Short=SMA_Short
        self.SMA_Long=SMA_Long
        self.start=start
        self.end=end
        self.results=None
        self.get_df()
        
        ##### Market Data #######
    def get_df(self):
        stickers=yf.download(self.symbol,start=self.start,end=self.end)
        df=stickers.Close.to_frame()
        df["B & H"]=np.log(df.Close.div(df.Close.shift(1)))
        df["SMA_Short"]=df.Close.rolling(self.SMA_Short).mean()
        df["SMA_Long"]=df.Close.rolling(self.SMA_Long).mean()
        df.dropna(inplace=True)
        self.df2=df
        
        return df
        
    ##### Backtesting Strategies #######
    def test_results(self):
        df=self.df2.copy().dropna()
        df["Position_Strat_1"]=np.where(df["SMA_Short"]>df["SMA_Long"],1,-1)
        df["Position_Strat_2"]=np.where(df["SMA_Short"]>df["SMA_Long"],1,0)
        df["Strat_1"]=df["B & H"]*df.Position_Strat_1.shift(1)
        df["Strat_2"]=df["B & H"]*df.Position_Strat_2.shift(1)
        df.dropna(inplace=True)
        df["Returns_B&H"]=df["B & H"].cumsum().apply(np.exp)
        df["Returns_Strat_1"]=df["Strat_1"].cumsum().apply(np.exp)
        df["Returns_Strat_2"]=df["Strat_2"].cumsum().apply(np.exp)
        perf=df["Returns_Strat_1"].iloc[-1]
        perf2=df["Returns_Strat_2"].iloc[-1]
        outperf=perf-df["Returns_B&H"].iloc[-1]
   
        self.results=df
        
        ret=np.exp(df["Strat_1"].sum())
        ret2=np.exp(df["Strat_2"].sum())
        std= df["Strat_1"].std()*np.sqrt(252)
        std2= df["Strat_2"].std()*np.sqrt(252)

        ### Print Ret/Std ###
        print('# Ret Strat 1: ', ret)
        print('# Ret  Strat 2: ', ret2)
        print('Std Strat 1: ', std)
        print('Std Strat 2: ', std2)





        # Cum Ret/Cum Max
        df['Cum_Ret'] = df['B & H'].cumsum().apply(np.exp)
        df['Cum_max'] = df.Cum_Ret.cummax()

        #Plot CUmSum
        df.Cum_Ret.plot(title = 'Buy and Hold')
        plt.show()

        #plot Cum max vs Cumsum
        df[['Cum_Ret', 'Cum_max']].plot()
        plt.show()

        ### Plot SMA VS Close ###
        df[['SMA_Short', 'SMA_Long', 'Close']].plot()
        plt.show()

        ### Plot SMA VS Position ###
        print('Rendement Strat2 VS B & H: ',df[['Strat_1', 'B & H']].sum())
        df[['SMA_Short', 'SMA_Long', 'Position_Strat_2']].plot(secondary_y='Position_Strat_2')
        plt.show()
    
        #return ret,std
        return round(perf,6), round(outperf,6), round(perf2,6)
    
    ##### Plot Strategies #######
    def plot_results(self):
        if self.results is None:
            print("Run the test please")
        else:
            title="{}| SMA_Short={} | SMA_Long{}".format(self.symbol,self.SMA_Short, self.SMA_Long)
            self.results[["Returns_B&H","Returns_Strat_2"]].plot(title=title, figsize=(12,8))
            plt.show()
        
    

Tester = BackTestingStrat('SPY', 50, 100, start='2010-01-01', end=dt.datetime.now())

Tester.test_results()
Tester.plot_results()
