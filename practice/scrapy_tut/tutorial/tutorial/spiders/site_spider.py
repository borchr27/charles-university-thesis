import scrapy
from tutorial.items import TutorialItem
import csv
import pathlib
import itertools

class SiteSpider(scrapy.Spider):
    name = 'site'
    def __init__(self, *args, **kwargs):
        super(SiteSpider).__init__(*args, **kwargs)
        p = pathlib.Path.cwd() / 'sites.csv'
        self.logger.warning('IS THE PATH TO OPEN site.csv CORRECT?...', p)
        with p.open() as file:
            data = list(itertools.chain.from_iterable(csv.reader(file)))
        self.start_urls = [url.strip() for url in data]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, self.parse)    

    def parse(self, response):
        item = TutorialItem()
        item['html'] = response.text
        item['name'] = response.url
        yield item
