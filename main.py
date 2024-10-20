from scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from scrapers.selenium_scraper import SeleniumScraper
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    # URL to scrape
    url = "https://medium.com/"

    # Using BeautifulSoupScraper
    bs_scraper = BeautifulSoupScraper()
    bs_soup = bs_scraper.scrape(url)

    if bs_soup:
        print("BeautifulSoupScraper result:")
        print(bs_soup.prettify())
    else:
        print("BeautifulSoupScraper failed to fetch the page.")

    # Using SeleniumScraper
    selenium_scraper = SeleniumScraper()
    selenium_page_source = selenium_scraper.scrape(url)

    if selenium_page_source:
        print("SeleniumScraper result:")
        print(selenium_page_source)
    else:
        print("SeleniumScraper failed to fetch the page.")


if __name__ == "__main__":
    main()