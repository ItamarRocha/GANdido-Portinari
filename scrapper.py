from datetime import datetime
import time
import urllib.request
import re
import requests
import numpy as np
import pandas as pd
import os

def download_and_save():
    
    for i in range(45,5197):
        link = f"http://www.portinari.org.br/img/sections/collection/artwork/200/{i}.jpg"
        try: 
            urllib.request.urlretrieve(link, f'{os.getcwd()}/data/{i}.jpg')
        except:
            print(f' ======   ERRO EM SALVAR A IMAGEM {i}   ======')
        
    return

download_and_save()