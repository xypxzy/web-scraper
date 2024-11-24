import readability
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from .base_analyzer import BaseAnalyzer
import spacy
import torch
import logging
from functools import lru_cache
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import readability
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextAnalyzer(BaseAnalyzer):
    def __init__(self, analysis_depth='advanced'):
        super().__init__()
        logger.info("Initializing text analysis pipelines...")

        # Analysis depth (basic, advanced)
        self.analysis_depth = analysis_depth
        # Detect device (CPU or GPU)
        self.device = 0 if torch.cuda.is_available() else -1

        # Sentiment Analysis Model
        try:
            # Load the sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model='cardiffnlp/twitter-roberta-base-sentiment',
                tokenizer='cardiffnlp/twitter-roberta-base-sentiment',
                device=self.device
            )
            logger.info("Sentiment analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise

        # T5 Model for Contextual Analysis
        try:
            # Load T5 model for contextual analysis
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
            logger.info("T5 contextual analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize T5 model: {e}")
            raise

        # spaCy Models for lemmatization
        try:
            self.nlp_en = spacy.load("en_core_web_md")
            self.loaded_spacy_models = {}
            logger.info("spaCy English model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy model: {e}")
            raise

        # Semantic Similarity Model
        try:
            self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
            logger.info("Semantic similarity model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize semantic similarity model: {e}")
            raise

    @lru_cache
    def get_spacy_model(self, language):
        """Load spaCy model for the specified language."""
        if language == "en":
            return self.nlp_en
        if language not in self.loaded_spacy_models:
            try:
                model_name = f"{language}_core_news_md"
                self.loaded_spacy_models[language] = spacy.load(model_name)
            except Exception as e:
                logger.error(f"Failed to load spaCy model for language '{language}': {e}")
                raise
        return self.loaded_spacy_models[language]

    def analyze_sentiment(self, text):
        """Analyzes the sentiment of the given text."""
        try:
            if not text.strip():
                return [{"label": "NEUTRAL", "score": 1.0}]
            # Handle neutral sentiments
            sentiment = self.sentiment_analyzer(text, return_all_scores=True)
            results = []

            for res in sentiment:
                # Find the label with the highest score
                max_score = max(res, key=lambda x: x['score'])
                results.append({"label": max_score['label'], "score": max_score['score']})
            return results
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return [{"label": "ERROR", "score": 0.0}]

    def analyze_context(self, text, task="summarize"):
        """Performs contextual analysis using T5."""
        try:
            if task == "summarize":
                input_text = f"summarize: {text}"
            elif task == "analyze sentiment":
                input_text = f"analyze sentiment: {text}"
            else:
                input_text = f"{task}: {text}"

            tokens = self.t5_tokenizer.encode(input_text, return_tensors="pt").to(self.t5_model.device)
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
            return "Context analysis failed."

    def lemmatize_and_normalize(self, text, language="en", custom_stop_words=None):
        """Performs lemmatization and normalization of text based on the language."""
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
            return text

    def analyze_keywords(self, text, language="en", max_keywords=10, min_score=0.3):
        """Extracts and scores keywords from the text."""
        try:
            # Preprocess text
            preprocessed_text = self.lemmatize_and_normalize(text, language)

            # Initialize the KeyBERT model
            kw_model = KeyBERT('all-mpnet-base-v2')

            # Extract keywords using multiple methods
            keywords = kw_model.extract_keywords(
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
            return []

    def analyze_text_structure(self, text):
        """Analyzes the structure of the text and provides insights."""
        try:
            nlp = self.nlp_en
            doc = nlp(text)

            # Calculate average sentence length and syntactic complexity
            sentences = list(doc.sents)
            avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)

            # Syntactic complexity (average number of clauses per sentence)
            total_clauses = sum(len([token for token in sent if token.dep_ == 'ROOT']) for sent in sentences)
            avg_clauses_per_sentence = total_clauses / len(sentences)

            # Passive voice detection
            passive_sentences = [sent for sent in sentences if any(token.dep_ == 'auxpass' for token in sent)]
            passive_voice_ratio = len(passive_sentences) / len(sentences)

            # Repetition analysis
            tokens = [token.text.lower() for token in doc if token.is_alpha]
            unique_tokens = set(tokens)
            repetition_ratio = (len(tokens) - len(unique_tokens)) / len(tokens)

            return {
                "avg_sentence_length": avg_sentence_length,
                "avg_clauses_per_sentence": avg_clauses_per_sentence,
                "passive_voice_ratio": passive_voice_ratio,
                "repetition_ratio": repetition_ratio,
                "total_sentences": len(sentences)
            }
        except Exception as e:
            logger.error(f"Error analyzing text structure: {e}")
            return {}

    def analyze_emotions(self, text):
        """Detects emotions in the given text."""
        try:
            # Use a model that supports multi-label emotion detection
            emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=self.device
            )
            emotions = emotion_analyzer(text)
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
            return {"ERROR": 1.0}

    def extract_text(self, soup):
        """Extracts text from HTML content."""
        try:
            # Advanced parsing using readability-lxml
            import readability
            doc = readability.Document(str(soup))
            full_text = doc.summary()

            # Parse the extracted summary
            from bs4 import BeautifulSoup
            summary_soup = BeautifulSoup(full_text, 'html.parser')
            paragraphs = [p.get_text(strip=True) for p in summary_soup.find_all('p')]
            headings = [h.get_text(strip=True) for h in summary_soup.find_all(re.compile('^h[1-6]$'))]

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

    def analyze_text_complexity(self, text):
        """Calculates readability and grade level of the text."""
        try:
            # Use multiple readability metrics
            r = readability.getmeasures(text)
            flesch_reading_ease = r['readability grades']['FleschReadingEase']
            flesch_kincaid_grade = r['readability grades']['FleschKincaidGradeLevel']
            gunning_fog = r['readability grades']['GunningFogIndex']
            smog = r['readability grades']['SMOGIndex']
            coleman_liau = r['readability grades']['ColemanLiauIndex']
            automated_readability = r['readability grades']['AutomatedReadabilityIndex']

            return {
                "flesch_reading_ease": flesch_reading_ease,
                "flesch_kincaid_grade": flesch_kincaid_grade,
                "gunning_fog_index": gunning_fog,
                "smog_index": smog,
                "coleman_liau_index": coleman_liau,
                "automated_readability_index": automated_readability
            }
        except Exception as e:
            logger.error(f"Error calculating text complexity: {e}")
            return {}

    def semantic_similarity(self, text, reference_keywords):
        """Calculates the semantic similarity between the text and reference keywords."""
        try:
            # Detailed similarity mapping
            text_embedding = self.semantic_model.encode([text], convert_to_tensor=True)
            keyword_embeddings = self.semantic_model.encode(reference_keywords, convert_to_tensor=True)
            similarities = util.cos_sim(text_embedding, keyword_embeddings)[0]
            similarity_scores = dict(zip(reference_keywords, similarities.tolist()))
            mean_similarity = similarities.mean().item()
            return {
                "similarity_scores": similarity_scores,
                "mean_similarity": mean_similarity
            }
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return {"similarity_scores": {}, "mean_similarity": 0.0}

    def rewrite_text(self, text, mode="simplify"):
        """Generates an alternative version of the text using T5."""
        try:
            if mode == "simplify":
                task_prefix = "simplify:"
            elif mode == "formalize":
                task_prefix = "paraphrase in a formal style:"
            elif mode == "summarize":
                task_prefix = "summarize:"
            elif mode == "expand":
                task_prefix = "elaborate on:"
            elif mode == "paraphrase":
                task_prefix = "paraphrase:"
            else:
                task_prefix = f"{mode}:"

            input_text = f"{task_prefix} {text}"
            tokens = self.t5_tokenizer.encode(input_text, return_tensors="pt").to(self.t5_model.device)
            rewritten_ids = self.t5_model.generate(
                tokens,
                max_length=512,
                min_length=50,
                num_beams=5,
                early_stopping=True
            )
            rewritten_text = self.t5_tokenizer.decode(rewritten_ids[0], skip_special_tokens=True)
            return rewritten_text
        except Exception as e:
            logger.error(f"Error rewriting text with T5: {e}")
            return "Text rewriting failed."

    def generate_recommendations(self, extracted_data, full_text, reference_keywords=None, user_goals=None):
        """
        Generates recommendations based on extracted analysis results.
        Automatically calls relevant analysis methods and combines results.
        """
        recommendations = [
            "---------- The text analysis is complete. Here are the recommendations based on the analysis ----------"
        ]

        try:
            # # Personalize recommendations based on user goals
            goals = user_goals if user_goals else ["general"]

            # Step 1: Analyze sentiment of headings
            heading_sentiments = self.analyze_sentiment(extracted_data['headings'])
            for i, (heading, sentiment) in enumerate(zip(extracted_data['headings'], heading_sentiments)):
                if sentiment['label'] != 'positive':
                    recommendations.append(
                        f"Consider improving the tone of the headline: '{heading}' (Sentiment: {sentiment['label']})"
                    )
                if len(heading.split()) > 10:
                    recommendations.append(f"The headline is too long: '{heading}'. Consider shortening it.")

            # Step 2: Analyze readability
            readability = self.analyze_text_complexity(full_text)
            if readability:
                grade_level = readability.get("flesch_kincaid_grade", 0)
                if grade_level > 8:
                    recommendations.append("The text is relatively complex. Simplify it for better engagement.")
            else:
                recommendations.append("Unable to calculate readability metrics.")

            # Step 3: Keyword analysis
            keywords = self.analyze_keywords(full_text)
            if len(keywords) < 5:
                recommendations.append("Add more relevant keywords to improve SEO visibility.")
            else:
                recommendations.append(f"Identified keywords: {', '.join([kw['keyword'] for kw in keywords])}")

            # Step 4: Text structure analysis
            text_structure = self.analyze_text_structure(full_text)
            if text_structure:
                if text_structure["avg_sentence_length"] > 20:
                    recommendations.append("Consider shortening your sentences to improve readability.")
                if text_structure["passive_voice_ratio"] > 0.2:
                    recommendations.append("Reduce the use of passive voice to make the text more engaging.")
                if text_structure["repetition_ratio"] > 0.1:
                    recommendations.append("Reduce word repetition to enhance the text's clarity.")

            # Step 5: Emotional analysis
            emotions = self.analyze_emotions(full_text)
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                if dominant_emotion not in ["joy", "trust"]:
                    recommendations.append(
                        f"The dominant emotion in the text is '{dominant_emotion}'. Consider adjusting the tone."
                    )

            # Step 6: Semantic similarity analysis
            if reference_keywords:
                similarity_results = self.semantic_similarity(full_text, reference_keywords)
                mean_similarity = similarity_results["mean_similarity"]
                if mean_similarity < 0.7:
                    recommendations.append(
                        f"The text has low semantic similarity ({mean_similarity:.2f}) with target keywords. Consider revising."
                    )
                else:
                    recommendations.append(
                        f"The text is semantically aligned with your target keywords (Similarity: {mean_similarity:.2f})."
                    )

            # Step 7: Generate alternative text suggestions
            rewritten_text = self.rewrite_text(full_text, mode="simplify")
            recommendations.append(f"Suggested simplified version of the text: {rewritten_text}")

            recommendations.append("---------- End of recommendations ----------")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("An error occurred during recommendation generation.")

        return recommendations
