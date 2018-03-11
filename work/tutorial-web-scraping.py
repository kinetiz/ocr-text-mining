#https://www.dataquest.io/blog/web-scraping-tutorial-python/

import pandas as pd
import requests
from bs4 import BeautifulSoup


## Tutorial
page = requests.get("http://dataquestio.github.io/web-scraping-pages/simple.html")
page
page.status_code
page.content

soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
list(soup)
list(soup.children)
html = list(soup.children)[2]
list(html.children)
body = list(html.children)[3]

list(body.children) # Return list as object, so require list() to display
p = list(body.children)[1]
p.get_text() #Extract text from tag <>

#####

soup = BeautifulSoup(page.content, 'html.parser')
p_tags = soup.find_all('p') # find all specific tags and retrive as list
a = [item for item in p_tags]
a[0]
soup.find('p').get_text() # find first tag

### Advance
page = requests.get("http://dataquestio.github.io/web-scraping-pages/ids_and_classes.html")
soup = BeautifulSoup(page.content, 'html.parser')
soup
allTagsP = soup.find_all('p', class_='outer-text')
allTags = soup.find_all(class_="outer-text")
soup.find_all(id="first")

## Css selector
# p a — finds all a tags inside of a p tag.
# body p a — finds all a tags inside of a p tag inside of a body tag.
# html body — finds all body tags inside of an html tag.
# p.outer-text — finds all p tags with a class of outer-text.
# p#first — finds all p tags with an id of first.
# body p.outer-text — finds any p tags with a class of outer-text inside of a body tag.

soup.select("div p")


##### Weather extraction

page = requests.get("http://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168")
soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id="seven-day-forecast")
forecast_items = seven_day.find_all(class_="tombstone-container")
tonight = forecast_items[0]
print(tonight.prettify())

period = tonight.find(class_="period-name").get_text()
short_desc = tonight.find(class_="short-desc").get_text()
temp = tonight.find(class_="temp").get_text()

print(period)
print(short_desc)
print(temp)

img = tonight.find("img")
desc = img['title'] # acess attribute 
alt = img['alt']
print(desc)
print(alt)

### Loop get all text from class -> selector is more powerful!
period_tags = seven_day.select(".tombstone-container .period-name")
periods = [pt.get_text() for pt in period_tags]
short_descs = [sd.get_text() for sd in seven_day.select(".tombstone-container .short-desc")]
temps = [t.get_text() for t in seven_day.select(".tombstone-container .temp")]
descs = [d["title"] for d in seven_day.select(".tombstone-container img")]

print(periods)
print(short_descs)
print(temps)
print(descs)

### Put in dataframe
weather = pd.DataFrame({
        "period": periods, 
        "short_desc": short_descs, 
        "temp": temps, 
        "desc":descs
    })
weather

weather["temp"]
temp_nums = weather["temp"].str.extract("(?P<temp_num>\d+)", expand=False)
weather["temp_num"] = temp_nums.astype('int')
temp_nums
weather["temp_num"].mean()

is_night = weather["temp"].str.contains("Low")
weather["is_night"] = is_night
is_night
