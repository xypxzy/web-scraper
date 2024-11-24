from analyzers.html_sctructure_analyzer import HTMLStructureAnalyzer
from analyzers.seo_analyzer import SEOAnalyzer
from analyzers.text_analyzer import TextAnalyzer
from db.db_handler import DBHandler
from scrapers.beautifulsoup_scraper import BeautifulSoupScraper
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    # URL to scrape
    url = "https://dastan-test.goldkeeper.net/html/phantom/index.html"
    # url = "https://world.hey.com/dhh/to-the-crazy-ones-e43822c7?ref=dailydev"

    # TODO: по сути нужно использовать только один,
    #  потому что один для динамического контента, а другой для статического.
    #  Поэтому нужно написать фунцию, которая будет определять, какой из них использовать.

    # Using BeautifulSoupScraper
    bs_scraper = BeautifulSoupScraper()
    html_content = bs_scraper.scrape(url)

    if html_content:
        logging.info("Page loaded successfully.")

        db_handler = DBHandler()

        try:
            # HTML structure analyzer
            structure_analyzer = HTMLStructureAnalyzer()
            structure_results = structure_analyzer.analyze(html_content, url)

            elements = structure_results['elements']
            recommendations = structure_results['recommendations']
            issues = structure_results['issues']

            # Text analysis
            text_analyzer = TextAnalyzer()
            extracted_texts = text_analyzer.extract_text(html_content)
            full_text = extracted_texts['full_text']
            reference_keywords = text_analyzer.analyze_keywords(full_text)

            text_recommendations = text_analyzer.generate_recommendations(extracted_texts, full_text, reference_keywords)

            recommendations.extend(text_recommendations)

            # SEO analysis
            seo_analyzer = SEOAnalyzer()
            seo_recommendations = seo_analyzer.analyze(elements)
            recommendations.extend(seo_recommendations)

            db_handler.save_analysis(url, recommendations, elements)

            print("\nRecommendations:")
            for recommendation in recommendations:
                print(f"- {recommendation}")

            print("\nIssues:")
            for issue in issues:
                print(f"- {issue}: {issues[issue]}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")

        finally:
            db_handler.close()

    else:
        logging.error("Failed to load the page.")

    # Using SeleniumScraper
    # selenium_scraper = SeleniumScraper()
    # process_scraper(selenium_scraper, url)
    # selenium_scraper.close()


if __name__ == "__main__":
    main()
