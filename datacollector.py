import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError
import json
import os

class GraphSpider(scrapy.Spider):
    name = "graph_spider"
    
    custom_settings = {
        # --- POLITENESS & CONFIG ---
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'ResearchBot/1.0 (+https://github.com/FaizalJnu)', 
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 0.5,
        'COOKIES_ENABLED': False,
        
        # --- FAST FAILURE & ERROR HANDLING ---
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 1,
        'DOWNLOAD_TIMEOUT': 10,
        'DOWNLOAD_FAIL_ON_DATALOSS': False,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 522, 524, 408, 429],
        
        # --- DEPTH CONTROL ---
        'DEPTH_LIMIT': 3, 
    }

    def start_requests(self):
        seeds = []
        try:
            with open('seeds.json', 'r') as f:
                seeds = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"CRITICAL ERROR: Could not load seeds.json: {e}")
            return

        for entry in seeds:
            if isinstance(entry, dict):
                url = entry.get('url')
                category = entry.get('category', 'unknown')
            else:
                url = entry
                category = 'unknown'

            if url:
                yield scrapy.Request(
                    url=url, 
                    callback=self.parse,
                    errback=self.handle_failure,
                    meta={'category': category, 'depth': 0} 
                )

    def clean_link_text(self, text):
        if not text:
            return None
            
        # 1. If text looks like a JSON blob (starts with {), try to extract description
        if text.strip().startswith('{') and "schema.org" in text:
            try:
                data = json.loads(text)
                return data.get('description', 'Image/Logo') 
            except:
                return None # If it's broken JSON, just ignore it
                
        # 2. Remove newlines and excess whitespace
        clean_text = " ".join(text.split()).strip()
        
        # 3. Filter out navigation noise and short nonsense
        noise_terms = ["skip navigation", "log in", "read more", "click here", "menu", "sign up"]
        if clean_text.lower() in noise_terms or len(clean_text) < 3:
            return None # Mark as ignore

        return clean_text

    def parse(self, response):
        # 1. Capture the Reward Signal (Cleaned Body Text)
        page_text = " ".join(response.css('body ::text').getall()).strip()
        page_text = " ".join(page_text.split()) 

        # 2. Capture Action Space (Links + CLEANED Text)
        outgoing_actions = []
        try:
            le = LinkExtractor(unique=True, allow_domains=[], deny_extensions=scrapy.linkextractors.IGNORED_EXTENSIONS)
            links = le.extract_links(response)
            
            for link in links:
                # Apply your new cleaning function here
                cleaned_text = self.clean_link_text(link.text)
                
                # Only add if it passed the filter (not None) and is http
                if link.url.startswith('http') and cleaned_text:
                    outgoing_actions.append({
                        'url': link.url,
                        'text': cleaned_text 
                    })
        except Exception as e:
            self.logger.warning(f"Link extraction error on {response.url}: {e}")

        # 3. Retrieve Metadata
        current_depth = response.meta.get('depth', 0)
        category = response.meta.get('category', 'unknown')

        # 4. Save to JSONL
        if outgoing_actions or len(page_text) > 100:
            yield {
                'url': response.url,
                'category': category,
                'depth': current_depth,
                'page_text_content': page_text[:5000], 
                'outgoing_links': outgoing_actions,
            }

        # 5. Recursion
        if current_depth < self.custom_settings['DEPTH_LIMIT']:
            for action in outgoing_actions:
                yield response.follow(
                    action['url'],
                    callback=self.parse,
                    errback=self.handle_failure,
                    meta={'category': category, 'depth': current_depth + 1} 
                )

    # --- ERROR HANDLER ---
    def handle_failure(self, failure):
        """
        This function is called if the request fails (404, 500, DNS error, Timeout).
        It logs the error cleanly and allows the spider to continue.
        """
        # log all failures
        self.logger.warning(f"Request failed: {failure.request.url}")

        if failure.check(HttpError):
            # These are non-200 responses (e.g. 404, 403, 500)
            response = failure.value.response
            self.logger.warning(f"HttpError on {response.url}: Status {response.status}")

        elif failure.check(DNSLookupError):
            # The URL is probably wrong or the site is down
            request = failure.request
            self.logger.warning(f"DNSLookupError on {request.url}")

        elif failure.check(TimeoutError, TCPTimedOutError):
            request = failure.request
            self.logger.warning(f"TimeoutError on {request.url}")

        else:
            # Some other weird error
            self.logger.warning(f"Unknown Error on {failure.request.url}: {failure.value}")


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(settings={
        "FEEDS": {
            "crawled_data.jsonl": {"format": "jsonlines", "overwrite": True},
        },
        "LOG_LEVEL": "INFO", 
    })
    
    process.crawl(GraphSpider)
    process.start()