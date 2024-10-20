import requests
from bs4 import BeautifulSoup
from .base_scraper import BaseScraper

class BeautifulSoupScraper(BaseScraper):
    def scrape(self, url):
        self.log_info(f"Fetching page: {url}")
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            return soup
        except requests.RequestException as e:
            self.log_error(f"Error fetching page: {e}")
            return None
