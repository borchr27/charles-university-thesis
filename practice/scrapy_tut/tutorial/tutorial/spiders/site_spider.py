import scrapy
from tutorial.items import TutorialItem

class QuoteSpider(scrapy.Spider):
    name = 'site'
    # allowed_domains = ['livedale.co.uk/']
    page = 1
    start_urls = ['https://www.livedale.co.uk/', 'http://mcgeoughsnisa.co.uk/']
    
    def parse(self, response):
        info = []
        url = 'https://www.livedale.co.uk/'
        item = TutorialItem()
        item['html'] = response.text
        item['name'] = response.url
        yield item
        yield scrapy.Request(url=url, callback=self.parse)
