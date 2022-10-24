# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import re
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from bs4 import BeautifulSoup


class ProcessingPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        if adapter.get('html'):
            html = adapter['html']
            soup = BeautifulSoup(html , 'html.parser')
            text = soup.get_text()  # get text from html
            text = " ".join(text.split()) # remove extra spaces
            adapter['html'] = text

            if adapter.get('name'):
                name = str(adapter['name']) 
                name = re.sub(r'[^\w]', '', name)   # remove special characters
                adapter['name'] = name.replace('https', '').replace('http', '').replace('www.', '') # remove https, http, www
                return item            
        else:
            raise DropItem(f"Missing website data at: {item}")

class SavePipeline:
    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        filename = 'site_' + adapter['name'] + '.txt'
        self.file = open(filename, 'w')
        if adapter.get('html'):
            text = adapter['html']
            self.file.write(text)
        return item