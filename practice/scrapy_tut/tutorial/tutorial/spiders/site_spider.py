import scrapy
from tutorial.items import TutorialItem

class SiteSpider(scrapy.Spider):
    name = 'site'
    with open('output.txt', 'w') as file:
        file.write("")

    def start_requests(self):
        yield scrapy.Request(self.url)

    def parse(self, response):
        item = TutorialItem()
        item['text'] = response.text
        item['v_url'] = response.url
        yield item