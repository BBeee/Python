import time
import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis',port=6379)

def get_hit_count():
	retries=5
	while True:
		try:
			return cache.incr('hit')
		except redis.exception.ConnectionError as exec:
			if retries == 0:
				raise exec
			retries -= 1
			time.sleep(1)

@app.route('/')
def hello():
	count = get_hit_count()
	return "Hi, I am J, and I've seen this {} times".format(count)
