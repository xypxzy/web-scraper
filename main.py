from analyzers.content_analyzer import ContentAnalyzer
from analyzers.html_sctructure_analyzer import HTMLStructureAnalyzer
from analyzers.seo_analyzer import SEOAnalyzer
from logs.process_scraper import process_scraper
from scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from scrapers.selenium_scraper import SeleniumScraper
import logging

def main():
    logging.basicConfig(level=logging.INFO)

    # URL to scrape
    url = "https://dastan-test.goldkeeper.net/html/phantom/index.html"

    # TODO: по сути нужно использовать только один,
    #  потому что один для динамического контента, а другой для статического.
    #  Поэтому нужно написать фунцию, которая будет определять, какой из них использовать.

    # Using BeautifulSoupScraper
    bs_scraper = BeautifulSoupScraper()
    html_content = bs_scraper.scrape(url)

    if html_content:
        logging.info("Page loaded successfully.")

        # Initialize the analyzer
        analyzer = HTMLStructureAnalyzer()
        analysis_results = analyzer.analyze(html_content)

        elements = analysis_results['elements']
        recommendations = analysis_results['recommendations']

        # SEO analysis
        seo_analyzer = SEOAnalyzer()
        seo_recommendations = seo_analyzer.analyze(elements)
        recommendations.extend(seo_recommendations)

        # Content analysis
        content_analyzer = ContentAnalyzer()
        content_recommendations = content_analyzer.analyze(elements)
        recommendations.extend(content_recommendations)

        # Print the results
        print("Elements:")
        for key, value in elements.items():
            print(f"\n{key.upper()}:")
            for item in value:
                print(f"- {item}")

        print("\nRecommendations:")
        for recommendation in recommendations:
            print(f"- {recommendation}")
        logging.error("Error loading page.")

    # Using SeleniumScraper
    # selenium_scraper = SeleniumScraper()
    # process_scraper(selenium_scraper, url)
    # selenium_scraper.close()

if __name__ == "__main__":
    main()