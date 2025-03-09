import pandas as pd

class WikipediaStockListing:
    def sp500(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S&P_500_companies'
        df = pd.read_html(url, header=0)[0]
        cols_ren = {'Security': 'Name', 'Ticker symbol': 'Symbol', 'GICS Sector': 'Sector',
                    'GICS Sub-Industry': 'Industry'}
        df = df.rename(columns=cols_ren)
        df = df[['Symbol', 'Name', 'Sector', 'Industry']]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df

    def sp100(self):
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
        df = pd.read_html(url, header=0)[2]
        cols_ren = {'Security': 'Name', 'Ticker symbol': 'Symbol', 'GICS Sector': 'Sector'}
        df = df.rename(columns=cols_ren)
        df = df[['Symbol', 'Name', 'Sector']]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df