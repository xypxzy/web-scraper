from transformers import pipeline
from .base_analyzer import BaseAnalyzer
import re

class TextAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        # Load the sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english'
        )

    def extract_text(self, soup):
        """Extracts the text from a string."""
        headings = [tag.get_text(strip=True) for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        paragraphs = [tag.get_text(strip=True) for tag in soup.find_all('p')]

        full_text = ' '.join(headings + paragraphs)
        return {
            'headings': headings,
            'paragraphs': paragraphs,
            'full_text': full_text
        }

    def analyze_sentiment(self, text):
        """Analyses the sentiment of a text."""
        sentiment = self.sentiment_analyzer(text)
        return sentiment

    def analyze_keywords(self, text, max_keywords=10):
        """Analyses the keywords in a text."""
        from keybert import KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english',
                                             top_n=max_keywords)
        return [kw[0] for kw in keywords]

    def readability_score(self, text):
        """Calculates the readability score of a text.(Flesch-Kincaid)"""
        from textstat import flesch_reading_ease
        score = flesch_reading_ease(text)
        return score

    def generate_recommendations(self, extracted_data, sentiment_results, readability, keywords):
        """Generates recommendations based on the text."""
        recommendations = []

        # Check for negative sentiment
        for i, sentiment in enumerate(sentiment_results):
            if sentiment['label'] != 'POSITIVE':
                recommendations.append(
                    f"Consider improving the tone of the headline: '{extracted_data['headings'][i]}'")

        # Check for readability
        if readability < 60:
            recommendations.append("The text is somewhat difficult to read. Consider simplifying it.")

        # Check for keywords
        if len(keywords) < 5:
            recommendations.append("Consider adding more relevant keywords to improve SEO.")

        return recommendations
