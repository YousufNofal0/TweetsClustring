import pandas as pd
import numpy as np
import re

#Cleansing
url = r'Tweets\bbchealth.txt'

df = pd.read_table(url, names = ['Tweet'])

df = df['Tweet'].str.split('|', expand=True)

df = df[[2]]
match = re.compile(r'http:\S+')
df[2] = df[2].str.replace(match, '', regex = True)

match = re.compile(r'@\S+')
df[2] = df[2].str.replace(match, '', regex = True)

match = re.compile(r'#')
df[2] = df[2].str.replace(match, '', regex = True)

df[2] = df[2].str.lower()



