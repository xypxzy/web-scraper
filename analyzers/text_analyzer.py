import logging
import re
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from bs4 import BeautifulSoup
from keybert import KeyBERT
from readability import Document as ReadabilityDocument  # For content extraction
from sentence_transformers import SentenceTransformer, util
from spacy.language import Language
import spacy
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline
)
import textstat  # For readability metrics

from analyzers.base_analyzer import BaseAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationStrategy(Enum):
    DEFAULT = 1
    AGGRESSIVE = 2
    CONSERVATIVE = 3


class TextAnalyzer(BaseAnalyzer):
    def __init__(self, analysis_depth: str = 'advanced'):
        super().__init__()
        logger.info("Initializing text analysis pipelines...")

        # Analysis depth (basic, advanced)
        self.analysis_depth = analysis_depth

        # Determine device (CPU or GPU)
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {'CUDA' if self.device_id >= 0 else 'CPU'}")

        # Initialize models
        try:
            # Sentiment Analysis Model
            self.sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model='cardiffnlp/twitter-roberta-base-sentiment',
                tokenizer='cardiffnlp/twitter-roberta-base-sentiment',
                device=self.device_id  # -1 for CPU, 0 for first GPU
            )
            logger.info("Sentiment analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise e

        try:
            # T5 Model for Contextual Analysis
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(self.torch_device)
            logger.info("T5 contextual analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize T5 model: {e}")
            raise e

        try:
            # spaCy Models for Lemmatization
            self.nlp_en = spacy.load("en_core_web_md")
            self.loaded_spacy_models = {}
            logger.info("spaCy English model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy model: {e}")
            raise e

        try:
            # Semantic Similarity Model
            self.semantic_model = SentenceTransformer('all-mpnet-base-v2').to(self.torch_device)
            logger.info("Semantic similarity model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize semantic similarity model: {e}")
            raise e

        try:
            # Emotion Analysis Model
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=self.device_id  # -1 for CPU, 0 for first GPU
            )
            logger.info("Emotion analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize emotion analyzer: {e}")
            raise e

        try:
            # Keyword Extraction Model
            self.kw_model = KeyBERT('all-mpnet-base-v2')
            logger.info("Keyword extraction model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize keyword extraction model: {e}")
            raise e

    def get_spacy_model(self, language: str) -> Language:
        """
        Loads the spaCy model for the specified language.

        Args:
            language (str): Language code (e.g., 'en' for English).

        Returns:
            spacy.Language: Loaded spaCy model.
        """
        if language == "en":
            return self.nlp_en
        if language not in self.loaded_spacy_models:
            try:
                model_name = f"{language}_core_news_md"
                self.loaded_spacy_models[language] = spacy.load(model_name)
                logger.info(f"spaCy model for language '{language}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load spaCy model for language '{language}': {e}")
                raise e
        return self.loaded_spacy_models[language]

    def analyze_sentiment(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyzes the sentiment of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing sentiment labels and scores.
        """
        try:
            if not text.strip():
                return [{"label": "NEUTRAL", "score": 1.0}]
            sentiment = self.sentiment_analyzer(text, return_all_scores=True)
            results = []

            for res in sentiment:
                # Find the label with the highest score
                max_score = max(res, key=lambda x: x['score'])
                results.append({"label": max_score['label'], "score": max_score['score']})
            return results
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise e

    def analyze_context(self, text: str, task: str = "summarize") -> str:
        """
        Performs contextual analysis using the T5 model.

        Args:
            text (str): The text to analyze.
            task (str): The task to perform ('summarize', 'analyze sentiment', etc.).

        Returns:
            str: The result of the contextual analysis.
        """
        try:
            if task == "summarize":
                input_text = f"summarize: {text}"
            elif task == "analyze sentiment":
                input_text = f"analyze sentiment: {text}"
            else:
                input_text = f"{task}: {text}"

            # Tokenize with truncation to ensure input does not exceed max length
            tokens = self.t5_tokenizer.encode(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.t5_tokenizer.model_max_length
            ).to(self.torch_device)

            summary_ids = self.t5_model.generate(
                tokens,
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            result = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return result
        except Exception as e:
            logger.error(f"Error analyzing context with T5: {e}")
            raise e

    def lemmatize_and_normalize(
        self,
        text: str,
        language: str = "en",
        custom_stop_words: Optional[List[str]] = None
    ) -> str:
        """
        Performs lemmatization and normalization of text based on the language.

        Args:
            text (str): The text to process.
            language (str): Language code.
            custom_stop_words (Optional[List[str]]): List of custom stop words.

        Returns:
            str: Lemmatized and normalized text.
        """
        try:
            nlp = self.get_spacy_model(language)
            doc = nlp(text)
            stop_words = nlp.Defaults.stop_words
            if custom_stop_words:
                stop_words = stop_words.union(set(custom_stop_words))

            lemmatized = [
                token.lemma_.lower() for token in doc
                if not token.is_stop and token.is_alpha and token.lemma_.lower() not in stop_words
            ]
            normalized_text = ' '.join(lemmatized)
            return normalized_text
        except Exception as e:
            logger.error(f"Error during lemmatization: {e}")
            raise e

    def analyze_keywords(
        self,
        text: str,
        language: str = "en",
        max_keywords: int = 10,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Extracts and scores keywords from the text.

        Args:
            text (str): The text to analyze.
            language (str): Language code.
            max_keywords (int): Maximum number of keywords.
            min_score (float): Minimum score threshold for a keyword.

        Returns:
            List[Dict[str, Any]]: List of keywords with their scores.
        """
        try:
            # Preprocess text
            preprocessed_text = self.lemmatize_and_normalize(text, language)

            # Extract keywords
            keywords = self.kw_model.extract_keywords(
                preprocessed_text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                use_maxsum=True,
                nr_candidates=20,
                top_n=max_keywords
            )

            # Filter by score
            filtered_keywords = [
                {"keyword": kw[0], "score": kw[1]}
                for kw in keywords
                if kw[1] >= min_score
            ]
            return filtered_keywords
        except Exception as e:
            logger.error(f"Error analyzing keywords: {e}")
            raise e

    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the structure of the text and provides insights.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, Any]: Data about the text structure.
        """
        try:
            nlp = self.nlp_en
            doc = nlp(text)

            # Calculate average sentence length and syntactic complexity
            sentences = list(doc.sents)
            if not sentences:
                return {
                    "avg_sentence_length": 0,
                    "avg_clauses_per_sentence": 0,
                    "passive_voice_ratio": 0,
                    "repetition_ratio": 0,
                    "total_sentences": 0
                }

            avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)

            # Syntactic complexity (average number of clauses per sentence)
            total_clauses = sum(
                len([token for token in sent if token.dep_ == 'ROOT']) for sent in sentences
            )
            avg_clauses_per_sentence = total_clauses / len(sentences) if len(sentences) > 0 else 0

            # Passive voice detection
            passive_sentences = [
                sent for sent in sentences if any(token.dep_ == 'auxpass' for token in sent)
            ]
            passive_voice_ratio = len(passive_sentences) / len(sentences) if len(sentences) > 0 else 0

            # Repetition analysis
            tokens = [token.text.lower() for token in doc if token.is_alpha]
            unique_tokens = set(tokens)
            repetition_ratio = (len(tokens) - len(unique_tokens)) / len(tokens) if len(tokens) > 0 else 0

            return {
                "avg_sentence_length": avg_sentence_length,
                "avg_clauses_per_sentence": avg_clauses_per_sentence,
                "passive_voice_ratio": passive_voice_ratio,
                "repetition_ratio": repetition_ratio,
                "total_sentences": len(sentences)
            }
        except Exception as e:
            logger.error(f"Error analyzing text structure: {e}")
            raise e

    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Detects emotions in the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, float]: Dictionary of emotions with their probabilities.
        """
        try:
            # Use the pipeline directly with text, allowing it to handle tokenization and device placement
            emotions = self.emotion_analyzer(
                text,
                truncation=True,       # Ensure text is truncated to the model's max length
                max_length=512         # Adjust based on the model's capabilities
            )
            # Aggregate emotions and their scores
            emotion_scores = defaultdict(float)
            for res in emotions:
                for emotion in res:
                    emotion_scores[emotion['label']] += emotion['score']
            # Normalize scores
            total_score = sum(emotion_scores.values())
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
            return dict(emotion_scores)
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            raise e

    def extract_text(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extracts text from HTML content.

        Args:
            soup (BeautifulSoup): BeautifulSoup object with HTML content.

        Returns:
            Dict[str, Any]: Extracted headings, paragraphs, and full text.
        """
        try:
            # Advanced parsing using readability-lxml
            doc = ReadabilityDocument(str(soup))
            full_text_html = doc.summary()

            # Parse the extracted summary
            summary_soup = BeautifulSoup(full_text_html, 'html.parser')
            paragraphs = [p.get_text(strip=True) for p in summary_soup.find_all('p')]
            headings = [
                h.get_text(strip=True)
                for h in summary_soup.find_all(re.compile('^h[1-6]$'))
            ]

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
            raise e

    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Calculates readability and grade level of the text.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, float]: Readability metrics.
        """
        try:
            # Use textstat for readability metrics
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog_index": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "automated_readability_index": textstat.automated_readability_index(text)
            }
        except Exception as e:
            logger.error(f"Error calculating text complexity: {e}")
            raise e

    def semantic_similarity(
            self,
            text: str,
            reference_keywords: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates the semantic similarity between the text and reference keywords.

        Args:
            text (str): The text to compare.
            reference_keywords (List[Dict[str, Any]]): List of reference keywords,
                each represented as a dictionary containing at least a 'keyword' key.

        Returns:
            Dict[str, Any]: A dictionary containing similarity scores for each keyword
                and the mean similarity score.
                Example:
                {
                    "similarity_scores": {
                        "Python": 0.85,
                        "programming": 0.78,
                        "language": 0.80
                    },
                    "mean_similarity": 0.81
                }
        """
        try:
            # Extract keyword strings from the list of dictionaries
            keywords = []
            for kw in reference_keywords:
                if isinstance(kw, dict):
                    keyword = kw.get('keyword')
                    if keyword and isinstance(keyword, str):
                        keywords.append(keyword)
                    else:
                        logger.warning(f"Invalid or missing 'keyword' in: {kw}")
                else:
                    logger.warning(f"Expected dict in reference_keywords, got: {type(kw)}")

            if not keywords:
                logger.error("No valid keywords found in reference_keywords.")
                raise ValueError("reference_keywords must contain at least one valid 'keyword' string.")

            # Detailed similarity mapping
            text_embedding = self.semantic_model.encode([text], convert_to_tensor=True)
            keyword_embeddings = self.semantic_model.encode(keywords, convert_to_tensor=True)
            similarities = util.cos_sim(text_embedding, keyword_embeddings)[0]

            # Convert similarities to a list of floats
            similarity_values = similarities.tolist()

            # Create a dictionary mapping each keyword to its similarity score
            similarity_scores = {keyword: score for keyword, score in zip(keywords, similarity_values)}

            # Calculate the mean similarity score
            mean_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 0.0

            return {
                "similarity_scores": similarity_scores,
                "mean_similarity": mean_similarity
            }

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            raise e

    def rewrite_text(self, text: str, mode: str = "simplify") -> str:
        """
        Generates an alternative version of the text using the T5 model.

        Args:
            text (str): The text to rewrite.
            mode (str): The mode of rewriting ('simplify', 'formalize', etc.).

        Returns:
            str: The rewritten text.
        """
        try:
            task_prefix = {
                "simplify": "simplify:",
                "formalize": "paraphrase in a formal style:",
                "summarize": "summarize:",
                "expand": "elaborate on:",
                "paraphrase": "paraphrase:"
            }.get(mode, f"{mode}:")  # Default to custom mode if not predefined

            input_text = f"{task_prefix} {text}"
            # Tokenize with truncation to ensure input does not exceed max length
            tokens = self.t5_tokenizer.encode(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.t5_tokenizer.model_max_length
            ).to(self.torch_device)

            rewritten_ids = self.t5_model.generate(
                tokens,
                max_length=512,
                min_length=50,
                num_beams=5,
                early_stopping=True
                # Removed 'truncation=True' as it's not a valid argument for generate
            )
            rewritten_text = self.t5_tokenizer.decode(rewritten_ids[0], skip_special_tokens=True)
            return rewritten_text
        except Exception as e:
            logger.error(f"Error rewriting text with T5: {e}")
            raise e

    def generate_recommendations(
        self,
        extracted_data: Dict[str, Any],
        full_text: str,
        reference_keywords: Optional[List[Dict[str, Any]]] = None,
        user_goals: Optional[List[str]] = None,
        strategy: RecommendationStrategy = RecommendationStrategy.DEFAULT
    ) -> List[str]:
        """
        Generates recommendations based on extracted analysis results and the selected strategy.

        Args:
            extracted_data (Dict[str, Any]): Extracted data (headings, paragraphs, etc.).
            full_text (str): Full text for analysis.
            reference_keywords (Optional[List[Dict[str, Any]]]): Reference keywords.
            user_goals (Optional[List[str]]): User goals.
            strategy (RecommendationStrategy): Selected recommendation strategy.

        Returns:
            List[str]: List of recommendations.
        """
        recommendations = [
            "---------- Text analysis complete. Here are the recommendations based on the analysis ----------"
        ]

        try:
            # Personalize recommendations based on user goals
            goals = user_goals if user_goals else ["general"]

            # Apply strategy-specific recommendations
            if strategy == RecommendationStrategy.DEFAULT:
                self._recommend_sentiment_analysis(extracted_data, recommendations)
                self._recommend_readability(full_text, recommendations)
                self._recommend_keywords(full_text, recommendations)
                self._recommend_text_structure(full_text, recommendations)
                self._recommend_emotions(full_text, recommendations)
                if reference_keywords:
                    self._recommend_semantic_similarity(full_text, reference_keywords, recommendations)
                self._recommend_rewritten_text(full_text, recommendations)
            elif strategy == RecommendationStrategy.AGGRESSIVE:
                # More stringent recommendations
                self._recommend_sentiment_analysis(extracted_data, recommendations)
                self._recommend_readability(full_text, recommendations)
                self._recommend_keywords(full_text, recommendations)
                self._recommend_text_structure(full_text, recommendations)
                self._recommend_emotions(full_text, recommendations)
                if reference_keywords:
                    self._recommend_semantic_similarity(full_text, reference_keywords, recommendations)
                # Additional aggressive recommendation
                recommendations.append("Consider a complete redesign of the text structure to enhance effectiveness.")
                self._recommend_rewritten_text(full_text, recommendations)
            elif strategy == RecommendationStrategy.CONSERVATIVE:
                # Less intrusive recommendations
                self._recommend_sentiment_analysis(extracted_data, recommendations)
                self._recommend_readability(full_text, recommendations)
                self._recommend_keywords(full_text, recommendations)
                # Limit to only a few recommendations
            else:
                logger.warning(f"Unknown recommendation strategy: {strategy}. Using DEFAULT strategy.")
                self._recommend_sentiment_analysis(extracted_data, recommendations)
                self._recommend_readability(full_text, recommendations)
                self._recommend_keywords(full_text, recommendations)
                self._recommend_text_structure(full_text, recommendations)
                self._recommend_emotions(full_text, recommendations)
                if reference_keywords:
                    self._recommend_semantic_similarity(full_text, reference_keywords, recommendations)
                self._recommend_rewritten_text(full_text, recommendations)

            recommendations.append("---------- End of recommendations ----------")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("An error occurred during recommendation generation.")
            raise e

        return recommendations

    def _recommend_sentiment_analysis(self, extracted_data: Dict[str, Any], recommendations: List[str]):
        """
        Generates recommendations based on sentiment analysis of headings.

        Args:
            extracted_data (Dict[str, Any]): Extracted data containing headings.
            recommendations (List[str]): List to append recommendations.
        """
        headings = extracted_data.get('headings', [])
        if headings:
            # Join headings into a single string for sentiment analysis
            heading_sentiments = self.analyze_sentiment('. '.join(headings))
            for heading, sentiment in zip(headings, heading_sentiments):
                if sentiment['label'].lower() != 'positive':
                    recommendations.append(
                        f"Consider improving the tone of the headline: '{heading}' (Sentiment: {sentiment['label']})"
                    )
                if len(heading.split()) > 10:
                    recommendations.append(f"The headline is too long: '{heading}'. Consider shortening it.")

    def _recommend_readability(self, full_text: str, recommendations: List[str]):
        """
        Generates recommendations based on readability metrics.

        Args:
            full_text (str): The full text to analyze.
            recommendations (List[str]): List to append recommendations.
        """
        readability_metrics = self.analyze_text_complexity(full_text)
        if readability_metrics:
            grade_level = readability_metrics.get("flesch_kincaid_grade", 0)
            if grade_level > 8:
                recommendations.append("The text is relatively complex. Simplify it for better reader engagement.")
        else:
            recommendations.append("Unable to calculate readability metrics.")

    def _recommend_keywords(self, full_text: str, recommendations: List[str]):
        """
        Generates recommendations based on keyword analysis.

        Args:
            full_text (str): The full text to analyze.
            recommendations (List[str]): List to append recommendations.
        """
        keywords = self.analyze_keywords(full_text)
        if len(keywords) < 5:
            recommendations.append("Add more relevant keywords to improve SEO visibility.")
        else:
            recommendations.append(f"Identified keywords: {', '.join([kw['keyword'] for kw in keywords])}")

    def _recommend_text_structure(self, full_text: str, recommendations: List[str]):
        """
        Generates recommendations based on text structure analysis.

        Args:
            full_text (str): The full text to analyze.
            recommendations (List[str]): List to append recommendations.
        """
        text_structure = self.analyze_text_structure(full_text)
        if text_structure:
            if text_structure["avg_sentence_length"] > 20:
                recommendations.append("Consider shortening your sentences to improve readability.")
            if text_structure["passive_voice_ratio"] > 0.2:
                recommendations.append("Reduce the use of passive voice to make the text more engaging.")
            if text_structure["repetition_ratio"] > 0.1:
                recommendations.append("Reduce word repetition to enhance the text's clarity.")

    def _recommend_emotions(self, full_text: str, recommendations: List[str]):
        """
        Generates recommendations based on emotional analysis.

        Args:
            full_text (str): The full text to analyze.
            recommendations (List[str]): List to append recommendations.
        """
        emotions = self.analyze_emotions(full_text)
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            if dominant_emotion.lower() not in ["joy", "trust"]:
                recommendations.append(
                    f"The dominant emotion in the text is '{dominant_emotion}'. Consider adjusting the tone."
                )

    def _recommend_semantic_similarity(self, full_text: str, reference_keywords: List[Dict[str, Any]], recommendations: List[str]):
        """
        Generates recommendations based on semantic similarity analysis.

        Args:
            full_text (str): The full text to analyze.
            reference_keywords (List[Dict[str, Any]]): Reference keywords.
            recommendations (List[str]): List to append recommendations.
        """
        similarity_results = self.semantic_similarity(full_text, reference_keywords)
        mean_similarity = similarity_results.get("mean_similarity", 0.0)
        if mean_similarity < 0.7:
            recommendations.append(
                f"The text has low semantic similarity ({mean_similarity:.2f}) with target keywords. Consider revising it."
            )
        else:
            recommendations.append(
                f"The text is semantically aligned with your target keywords (Similarity: {mean_similarity:.2f})."
            )

    def _recommend_rewritten_text(self, full_text: str, recommendations: List[str]):
        """
        Generates and appends a rewritten version of the text.

        Args:
            full_text (str): The full text to rewrite.
            recommendations (List[str]): List to append recommendations.
        """
        rewritten_text = self.rewrite_text(full_text, mode="simplify")
        recommendations.append(f"Suggested simplified version of the text: {rewritten_text}")

    def _choose_strategy(self, strategies: List[tuple]) -> RecommendationStrategy:
        """
        Selects a recommendation strategy based on provided weights.

        Args:
            strategies (List[tuple]): List of tuples containing (strategy, weight).

        Returns:
            RecommendationStrategy: Selected recommendation strategy.
        """
        strategy_names, weights = zip(*strategies)
        chosen_strategy = random.choices(strategy_names, weights=weights, k=1)[0]
        logger.info(f"Selected recommendation strategy: {chosen_strategy.name}")
        return chosen_strategy

    def generate_recommendations_with_ab_testing(
        self,
        extracted_data: Dict[str, Any],
        full_text: str,
        reference_keywords: Optional[List[Dict[str, Any]]] = None,
        user_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates recommendations using A/B testing to select different strategies.

        Args:
            extracted_data (Dict[str, Any]): Extracted data.
            full_text (str): The full text to analyze.
            reference_keywords (Optional[List[Dict[str, Any]]]): Reference keywords.
            user_goals (Optional[List[str]]): User goals.

        Returns:
            Dict[str, Any]: Generated recommendations along with the strategy used.
        """
        try:
            # Define strategies with their respective probabilities
            strategies = [
                (RecommendationStrategy.DEFAULT, 0.5),
                (RecommendationStrategy.AGGRESSIVE, 0.3),
                (RecommendationStrategy.CONSERVATIVE, 0.2)
            ]
            strategy = self._choose_strategy(strategies)

            recommendations = self.generate_recommendations(
                extracted_data,
                full_text,
                reference_keywords,
                user_goals,
                strategy=strategy
            )

            # Log the selected strategy
            logger.info(f"Used strategy {strategy.name} for generating recommendations.")

            return {
                "strategy": strategy.name,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"Error generating recommendations with A/B testing: {e}")
            return {
                "strategy": "ERROR",
                "recommendations": ["An error occurred during recommendation generation."]
            }
