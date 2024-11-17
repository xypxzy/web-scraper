from scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from scrapers.selenium_scraper import SeleniumScraper
import logging

def process_scraper(scraper, url):
    result = scraper.scrape(url)
    if result:
        print(f"{scraper.__class__.__name__} result:")
    else:
        print(f"{scraper.__class__.__name__} failed to fetch the page.")

def main():
    logging.basicConfig(level=logging.INFO)

    # URL to scrape
    url = "https://medium.com/"

    # Using BeautifulSoupScraper
    bs_scraper = BeautifulSoupScraper()
    process_scraper(bs_scraper, url)

    # Using SeleniumScraper
    selenium_scraper = SeleniumScraper()
    process_scraper(selenium_scraper, url)
    selenium_scraper.close()

if __name__ == "__main__":
    main()