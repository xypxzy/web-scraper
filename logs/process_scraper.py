def process_scraper(scraper, url):
    result = scraper.scrape(url)
    if result:
        print(f"{scraper.__class__.__name__} result:")
        print(result)
    else:
        print(f"{scraper.__class__.__name__} failed to fetch the page.")
