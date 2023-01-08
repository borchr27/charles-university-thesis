import scrapy
from tutorial.items import TutorialItem

class SiteSpider(scrapy.Spider):
    name = 'site'
    
    def start_requests(self):
        yield scrapy.Request(self.url)

    def parse(self, response):
        item = TutorialItem()
        item['text'] = response.text
        item['v_url'] = response.url
        yield item