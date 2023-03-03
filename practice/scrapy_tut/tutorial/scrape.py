import random
import subprocess
import csv
import pathlib
import time
import thesis_utils as tu
import logging

##########################################################################################
# This is the code that I used to scrape the data using the websites from the csv file 
# then calling scrapy to get the text from the website.

def scrape():
    # create database object, open connection 
    db = tu.PostgresDatabase()
    db.connect()
    p = pathlib.Path.cwd() / 'sites.csv'

    # get the data from the csv file and but each website data into a Website object and store in data list
    data = []
    with p.open() as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            w = tu.Website(id=None, name=row[0], i_url=row[1], v_url=None, text=None, category=row[2], tags=row[3], origin=None)
            data.append(w)
    
    assert w.origin != None, "Origin is None"

    # individual scrapy calls for each website object in data list
    filename = 'output.txt'
    for i in range(len(data)):
        # if the website is not a facebook page, call scrapy and get the site text
        if "facebook" not in data[i].i_url:
            try:
                subprocess.call([f"scrapy crawl site -a url='{data[i].i_url}'"], shell = True, cwd='./', timeout=10)
                with open(filename, 'r') as file:
                    reader = csv.reader(file)
                    input = next(reader)
                    data[i].v_url = input[0]
                    data[i].text = input[1]
                db.add_website(data[i])
                # choose a random time to sleep for between 5 and 10 seconds
                time.sleep(random.randint(7, 13))
            except Exception as e:
                logging.info(f"Error: (main.py) {e}")
                print(f"Scrapy call failed for {data[i].name}")
                continue
        else:
            print(f"Facebook url found for {data[i].name}")
            continue

    db.close()

if __name__ == '__main__':
    scrape()
