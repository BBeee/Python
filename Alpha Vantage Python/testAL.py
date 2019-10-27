import requests
response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=5min&outputsize=full&apikey=06NCM8Z6D0E5CNQ6")
dir(response)
print(response.json())
