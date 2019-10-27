import urllib.request
import json
import os

#mainURL that will be appended to return our requests
mainURL = "https://www.alphavantage.co/query?"
#API key that will be needed to authenticate
myKey = "06NCM8Z6D0E5CNQ6"


#loads a json object of based on what you input in dailyData(SYMBOL)
requestTypeURL = "function=TIME_SERIES_INTRADAY"
interval="interval=1min"
outputsize="outputsize=full"


def import_web(ticker):
    symbolURL = "symbol=" + str(ticker)
    apiURL = "apikey=" + myKey
    url =  mainURL + requestTypeURL + '&' + symbolURL + '&'  + interval + '&' + apiURL + '&outputsize=full&datatype=json'
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()
    return mystr


def get_value(ticker):
    js = import_web(ticker)
    parsed_data = json.loads(js) # loads the json and converts the json string into dictionary
    ps = parsed_data['Time Series (1min)']
    print(ps)

                
def main():
    #Start Process
    company_list = ['GOOGL','MSFT','ORCL','FB','AAPL','TSLA'];
    try:
        for company in company_list:
            print("Starting with " + company)
            get_value(company)
            print("Ended Writing Data of " + company)
    except Exception as e:
        print(e)

main()
