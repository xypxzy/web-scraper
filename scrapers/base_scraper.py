import logging

class BaseScraper:
    def __init__(self):
        """Initialize the scraper with a logger."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_error(self, message):
        """Log an error message."""
        self.logger.error(message)

    def log_info(self, message):
        """Log an informational message."""
        self.logger.info(message)

    def scrape(self, url):
        """Scrape the given URL.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
