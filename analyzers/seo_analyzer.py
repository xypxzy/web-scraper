from .base_analyzer import BaseAnalyzer

class SEOAnalyzer(BaseAnalyzer):
    def analyze(self, elements):
        """Analyses page elements from an SEO perspective."""
        self.logger.info("Starting SEO analysis")
        recommendations = []

        # TODO: Add more SEO checks and research SEO best practices...

        # Checking amount of internal and external links
        internal_links = [link for link in elements['links'] if self.is_internal_link(link['href'])]
        if len(internal_links) < 3:
            recommendations.append("Add more internal links to improve SEO.")

        return recommendations

    def is_internal_link(self, link):
        """Checks if a link is internal."""
        return link.startswith("/")