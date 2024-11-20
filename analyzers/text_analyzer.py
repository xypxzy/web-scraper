from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from .base_analyzer import BaseAnalyzer
import spacy
import torch
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()

        # Detect device (CPU or GPU)
        self.device = 0 if torch.cuda.is_available() else -1

        try:
            # Load the sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model='nlptown/bert-base-multilingual-uncased-sentiment',
                device=self.device
            )
            logger.info("Sentiment analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise

        try:
            # Load T5 model for contextual analysis
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            logger.info("T5 contextual analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize T5 model: {e}")
            raise

        try:
            # Load spaCy model for lemmatization
            self.nlp_en = spacy.load("en_core_web_sm")
            self.loaded_spacy_models = {}
            logger.info("spaCy English model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy model: {e}")
            raise

    @lru_cache
    def get_spacy_model(self, language):
        """Load spaCy model for the specified language."""
        if language == "en":
            return self.nlp_en
        if language not in self.loaded_spacy_models:
            try:
                self.loaded_spacy_models[language] = spacy.load(f"{language}_core_news_sm")
            except Exception as e:
                logger.error(f"Failed to load spaCy model for language '{language}': {e}")
                raise
        return self.loaded_spacy_models[language]

    def extract_text(self, soup):
        """Extracts text from HTML content."""
        try:
            # Remove scripts, styles, and comments
            for script_or_style in soup(["script", "style", "noscript"]):
                script_or_style.decompose()

            # Extract headings and paragraphs
            headings = [tag.get_text(strip=True) for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            paragraphs = [tag.get_text(strip=True) for tag in soup.find_all('p')]

            # Remove duplicates
            headings = list(dict.fromkeys(headings))
            paragraphs = list(dict.fromkeys(paragraphs))

            full_text = ' '.join(headings + paragraphs)
            return {
                'headings': headings,
                'paragraphs': paragraphs,
                'full_text': full_text
            }
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return {'headings': [], 'paragraphs': [], 'full_text': ''}

    def lemmatize_and_normalize(self, text, language="en"):
        """Performs lemmatization and normalization of text based on the language."""
        try:
            nlp = self.get_spacy_model(language)
            doc = nlp(text)
            lemmatized = [
                token.lemma_ for token in doc
                if not token.is_stop and token.is_alpha
            ]
            return ' '.join(lemmatized)
        except Exception as e:
            logger.error(f"Error during lemmatization: {e}")
            return text

    def analyze_context(self, text):
        """Performs contextual analysis using T5."""
        try:
            input_text = f"analyze sentiment: {text}"
            tokens = self.t5_tokenizer.encode(input_text, return_tensors="pt")
            summary_ids = self.t5_model.generate(tokens, max_length=100, min_length=5, length_penalty=2.0, num_beams=4)
            result = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return result
        except Exception as e:
            logger.error(f"Error analyzing context with T5: {e}")
            return "Context analysis failed."

    def analyze_sentiment(self, text):
        """Analyzes the sentiment of the given text."""
        try:
            if not text.strip():
                return [{"label": "NEUTRAL", "score": 1.0}]
            sentiment = self.sentiment_analyzer(text)
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return [{"label": "ERROR", "score": 0.0}]

    def analyze_keywords(self, text, max_keywords=10):
        """Extracts keywords from the given text."""
        from keybert import KeyBERT
        try:
            text = self.lemmatize_and_normalize(text)
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english',
                                                 top_n=max_keywords)
            return [{"keyword": kw[0], "score": kw[1]} for kw in keywords]
        except Exception as e:
            logger.error(f"Error analyzing keywords: {e}")
            return []

    def readability_score(self, text):
        """Calculates the readability score of the text (Flesch-Kincaid)."""
        from textstat import flesch_reading_ease
        try:
            score = flesch_reading_ease(text)
            return score
        except Exception as e:
            logger.error(f"Error calculating readability score: {e}")
            return 0

    def generate_recommendations(self, extracted_data, sentiment_results, readability, keywords):
        """Generates recommendations based on the extracted analysis."""
        recommendations = []

        try:
            # Recommendations for headings
            for i, (heading, sentiment) in enumerate(zip(extracted_data['headings'], sentiment_results)):
                if sentiment['label'] != 'POSITIVE':
                    recommendations.append(
                        f"Consider improving the tone of the headline: '{heading}' (Sentiment: {sentiment['label']})")
                if len(heading.split()) > 10:
                    recommendations.append(
                        f"The headline is too long: '{heading}'. Consider shortening it.")

            # Recommendations for readability
            if readability < 60:
                recommendations.append("The text is somewhat difficult to read. Simplify it for better user engagement.")

            # Recommendations for keywords
            if len(keywords) < 5:
                recommendations.append("Add more relevant keywords to improve SEO visibility.")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")

        return recommendations
