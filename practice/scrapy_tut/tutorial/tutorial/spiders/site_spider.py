import scrapy
from tutorial.items import TutorialItem

class SiteSpider(scrapy.Spider):
    name = 'site'
    def __init__(self, *args, **kwargs):
        super(SiteSpider).__init__(*args, **kwargs)
        self.start_urls = ['https://www.livedale.co.uk/', 'http://mcgeoughsnisa.co.uk/', 'http://bluebadgerwi.com/', 'https://m.facebook.com/tostoria', 'https://parcelbroker.co.uk/']

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, self.parse)    

    def parse(self, response):
        item = TutorialItem()
        item['html'] = response.text
        item['name'] = response.url
        yield item
