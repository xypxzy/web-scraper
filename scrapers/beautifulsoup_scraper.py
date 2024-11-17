import requests
from bs4 import BeautifulSoup
from .base_scraper import BaseScraper

class BeautifulSoupScraper(BaseScraper):
    def scrape(self, url):
        self.log_info(f"Fetching page: {url}")
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
        except requests.HTTPError as http_err:
            self.log_error(f"HTTP error occurred: {http_err}")
            return None
        except requests.ConnectionError as conn_err:
            self.log_error(f"Connection error occurred: {conn_err}")
            return None
        except requests.Timeout as timeout_err:
            self.log_error(f"Request timed out: {timeout_err}")
            return None
        except requests.RequestException as e:
            self.log_error(f"Error fetching page: {e}")
            return None
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            return soup
