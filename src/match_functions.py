import json
import pandas as pd
import re
import urllib.request

'''
This functions is to request a link
'''
def request_link(instance_address):
    with urllib.request.urlopen(instance_address) as url:
        instance = json.loads(url.read().decode()) #load the NRRT associated with the UOA for example the https://one
    return (instance['LINK'])