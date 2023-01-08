# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import re
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from bs4 import BeautifulSoup
import psycopg2
import numpy as np

class ProcessingPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        if adapter.get('text'):
            html = adapter['text']
            soup = BeautifulSoup(html , 'html.parser')
            text = soup.get_text()  # get text from html
            text = " ".join(text.split()) # remove extra spaces
            adapter['text'] = text
            return item      
        else:
            raise DropItem(f"Missing website data at: {item}")

class SavePipeline:
    def __init__(self) -> None:
        self.file = None
    
    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        filename = 'output.txt'
        self.file = open(filename, 'w')
        if adapter.get('text'):
            text = adapter['text']
            url = adapter['v_url']
            self.file.write(url + ",")
            self.file.write(text)
        else:
            self.file.write(",")
        return item