import scrapy
import json
from math import ceil

class QuoteSpider(scrapy.Spider):
    name = 'quote'
    allowed_domains = ['sreality.cz']
    page = 1
    start_urls = ['https://www.sreality.cz/api/en/v2/estates?category_main_cb=2&category_type_cb=1&page=1&per_page=20']

    
    def parse(self, response):
        info = []
        json_load = json.loads(response.text)
        data = json_load['_embedded']['estates']
        results_count = int(json_load['result_size'])

        with open("properties.txt", "a") as myfile:
            for x in data:
                # info.append(f"name: {x['name']}, image: {x['_links']['images'][0]['href']}")
                myfile.write(f"name: {x['name']}, image: {x['_links']['images'][0]['href']} \n")

        if (results_count > 200 and self.page < 11) or (results_count < 200 and self.page <= ceil(results_count/20)):
            self.page += 1
            url = f"https://www.sreality.cz/api/en/v2/estates?category_main_cb=2&category_type_cb={self.page}&page=1&per_page=20"
            yield scrapy.Request(url=url, callback=self.parse)
    