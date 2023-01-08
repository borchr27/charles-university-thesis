import subprocess
import csv
import pathlib
import postgres_db as postgres
from website import Website

# 1-45 are in the database done 1/8
# 46-100 are in the database done 1/8

if __name__ == '__main__':
    # create database object, open connection 
    db = postgres.PostgresDatabase()
    p = pathlib.Path.cwd() / 'sites.csv'

    # get the data from the csv file and but each website data into a Website object and store in data list
    data = []
    with p.open() as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            w = Website(name=row[0], i_url=row[1], v_url=None, text=None, category=row[2], tags=row[3])
            data.append(w)
    
    # individual scrapy calls for each website object in data list
    for i in range(len(data)):
        subprocess.call([f"scrapy crawl site -a url='{data[i].i_url}'"], shell = True, cwd='./')
        filename = 'output.txt'
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            input = next(reader)
            try:
                data[i].v_url = input[0]
                data[i].text = input[1]
            except:
                pass
        db.add_item(data[i])
