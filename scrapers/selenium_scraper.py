from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from .base_scraper import BaseScraper

class SeleniumScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def scrape(self, url):
        self.log_info(f"Loading page with Selenium: {url}")
        try:
            self.driver.get(url)
            return self.driver.page_source
        except Exception as e:
            self.log_error(f"Error loading page: {e}")
            return None
        finally:
            self.driver.quit()
