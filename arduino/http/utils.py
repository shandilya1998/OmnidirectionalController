import requests
import numpy as np

def make_get_request(url, params):
    return requests.get(url = url, params = params)

if __name__ == '__main__':
    url = 'http://192.168.1.15:80'
    params = {'sr1' : int(np.random.random() * 150 + 20)}
    response = make_get_request(url, params)
    print(response)
