import json
import urllib.request
import datetime

#mainURL that will be appended to return our requests
mainURL = "https://www.alphavantage.co/query?"
#API key that will be needed to authenticate
myKey = "06NCM8Z6D0E5CNQ6"

#For daily activity
'''
REQUEST NEEDS:
function: will ALWAYS be equal to 'TIME_SERIES_DAILY' for daily
symbol: user defined stock ticker
outputsize: default is 'compact' which only returns 100 data points, 
otherwise
we can define it as 'full' to get up to 20 years worth
datatype: default is 'json', but we can also request 'csv'
'''

#loads a json object of based on what you input in dailyData(SYMBOL)
requestTypeURL = "function=TIME_SERIES_INTRADAY"
interval="interval=5min"
outputsize="outputsize=full"

def dailyData(symbol, requestType=requestTypeURL, apiKey=myKey):
    symbolURL = "symbol=" + str(symbol)
    apiURL = "apikey=" + myKey
    completeURL = mainURL + requestType + '&' + symbolURL + '&'  + interval + '&'  + outputsize + '&' + apiURL
    req = urllib.request.urlopen(completeURL)
    #data = json.load(req)
    print(completeURL)
    return req


#making a json object for Apple and example of getting a date's activity 

apple = dailyData('AAPL')
print(apple.read())

