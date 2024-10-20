import logging

class BaseScraper:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_error(self, message):
        self.logger.error(message)

    def log_info(self, message):
        self.logger.info(message)

    def scrape(self, url):
        raise NotImplementedError("This method should be overridden by subclasses")
