import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.europarl.europa.eu/plenary/de/parliament-positions.html?tabActif=tabLast#sidesForm"
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

print(page.status_code)

