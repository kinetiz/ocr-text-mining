=============================================================================
### Description ###
# The cw is to perform data mining techniques to clean, extract, explore and present 
# insight/relationship of 24 ocr books which is unstructed html data. 
=============================================================================
### Steps
#- Scrape the text from html put in Dataframe
#- Perform feature extraction to transform document text to vectors
#- Perform multidimension scaling techniques to reduce dimensions to be ready for exploration
#- Visualise the data in low dim and perform clustering to find relationship btw each book
#- Find some insight!
=============================================================================
import os
import sys
import pandas as pd
from bs4 import BeautifulSoup as bs

###### Scrap html to dataframe
walk_dir = "G:\\work\\ocr-text-mining\\gap-html\\gap-html\\"

print('walk_dir = ' + walk_dir)
print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))

for root, subdirs, files in os.walk(walk_dir):
    print('--\nprocessing dir = ' + root)
    # os.path.join => join str with //
    for subdir in subdirs:
        print('\t- subdirectory ' + subdir)

    for filename in files:
        file_path = os.path.join(root, filename)

        print('\t- file %s (full path: %s)' % (filename, file_path))
        

                
soup = bs(open(file_path), "html.parser")           
soup
#########
