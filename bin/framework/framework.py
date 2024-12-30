import numpy as np
import torch

from collections import defaultdict
from nltk import ngrams
from nltk.stem import Cistem
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from bin.utils.types import DataSet

class Framework:

    """
    A class to represent a framework for evaluating text generation models.
    Attributes:
        cistem (Cistem): The Cistem stemmer.
    Methods:
        calculate_perplexity(text, model, tokenizer): Calculate the perplexity of a given text using BERT's masked language modeling.
        distinct_n(text, n): Calculate the distinct-n metric of a given text.
        calculate_coherence(text, model): Calculate the coherence of a given text using a sentence embedding model.
        type_token_ratio(text): Calculate the type-token ratio of a given text using the Cistem stemmer.
        moving_average_ttr(text, window_size): Calculate the moving average type-token ratio of a given text using the Cistem stemmer.
        bleu_score(hypothesis, reference): Calculate the BLEU score of a given hypothesis with respect to a reference.
        task_specific_performance(train_data, test_data, model): Calculate the performance of a model on a task-specific dataset.
        inter_sentence_similarity(sentences, model): Calculate the inter-sentence similarity of a list of sentences using a sentence embedding model.
        discourse_coherence(text, model): Calculate the discourse coherence of a given text using the entity grid model.
        apply_framework(): Calculate all metrics for a text generation model.
    """

    def __init__(self):
        self.cistem = Cistem()

    def calculate_perplexity(self, text, model=None, tokenizer=None):
        """
        Calculate the perplexity of a given text using BERT's masked language modeling.
        
        Args:
            text (str): The text to calculate the perplexity of.
            model (transformers.PreTrainedModel): The BERT model. By default, uses bert-base-uncased.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer. By default, uses bert-base-uncased.
        
        Returns:
            float: The perplexity of the text.
        """
        if model is None:
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        print(f"Loaded model {model.__class__.__name__} and tokenizer {tokenizer.__class__.__name__}")
        
        # Tokenize input text
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids
        
        # Create attention mask
        attention_mask = encodings.attention_mask
        
        # Calculate token likelihoods
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Get the log probability of each actual token
            token_log_probs = log_probs[0, range(len(input_ids[0])), input_ids[0]]
            
            # Get special tokens mask and convert to tensor
            special_tokens_mask = torch.tensor(
                tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True),
                dtype=torch.bool
            )
            
            # Invert mask to get non-special tokens
            non_special_tokens_mask = ~special_tokens_mask
            
            # Calculate perplexity only on non-special tokens
            non_special_token_log_probs = token_log_probs[non_special_tokens_mask]
            
            # Calculate mean negative log likelihood
            mean_neg_log_likelihood = -non_special_token_log_probs.mean().item()
            
            # Calculate perplexity
            perplexity = np.exp(mean_neg_log_likelihood)
        
        return perplexity.astype(float)

    def distinct_n(self, text, n):
        """
        Calculate the distinct-n metric of a given text.
        Args:
            text (str): The text to calculate the distinct-n of.
            n (int): The n-gram order.
        Returns:
            float: The distinct-n of the text.
        """
        tokens = text.split()
        n_grams = list(ngrams(tokens, n))
        return float(len(set(n_grams)) / len(n_grams))

    def type_token_ratio(self, text):
        """
        Calculate the type-token ratio of a given text using the Cistem stemmer.
        Args:
            text (str): The text to calculate the type-token ratio of.
        Returns:
            float: The type-token ratio of the text.
        """
        tokens = text.split()
        types = set([self.cistem.stem(token) for token in tokens])
        return len(types) / len(tokens)

    def moving_average_ttr(self, text, window_size=100):
        """
        Calculate the moving average type-token ratio of a given text using the Cistem stemmer.
        Args:
            text (str): The text to calculate the moving average type-token ratio of.
            window_size (int): The window size for the moving average.
        Returns:
            list: The moving average type-token ratio of the text.
        """
        ttrs = []
        for i in range(0, len(text), window_size):
            window = text[i:i+window_size]
            ttrs.append(self.type_token_ratio(window))
        
        moving_average = np.mean(ttrs)
        return moving_average

    def bleu_score(self, hypothesis: str, reference: str) -> float:
        """
        Calculate the BLEU score of a given hypothesis with respect to a reference.
        Args:
            hypothesis (str): The hypothesis text.
            reference (str): The reference text.
        Returns:
            float: The BLEU score of the hypothesis.
        """
        return sentence_bleu([reference.split()], hypothesis.split())

    def task_specific_performance(self, train_data: DataSet, test_data: DataSet, model = None):
        """
        Calculate the performance of a model on a task-specific dataset.
        Args:
            train_data (DataSet): The training dataset.
            test_data (DataSet): The test dataset.
            model (Model): The model to evaluate.
        Returns:
            float: The performance of the model on the test dataset.
        """
        # Train model on synthetic data
        model.train(train_data)
        # Evaluate on real test data
        results = model.evaluate(test_data)
        # Return relevant metrics (e.g., F1 score)
        metrics = results['metrics']
        return metrics['f1']

    def inter_sentence_similarity(self, sentences: list, model=None):
        """
        Calculate the inter-sentence similarity of a list of sentences using a sentence embedding model.
        Args:
            sentences (list): The list of sentences.
            model (sentence_transformers.SentenceTransformer): The sentence embedding model.
        Returns:
            float: The inter-sentence similarity of the sentences.
        """
        if not model:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not sentences or len(sentences) < 2:
            raise ValueError("At least two sentences are required to compute inter-sentence similarity.")
        
        # Generate embeddings
        embeddings = model.encode(sentences)
        
        # Ensure embeddings are 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Exclude diagonal (self-similarity) elements and compute the mean
        n = similarities.shape[0]
        sum_of_similarities = np.sum(similarities) - np.sum(np.diag(similarities))  # Exclude diagonal
        num_comparisons = n * (n - 1)  # Total number of off-diagonal elements
        
        inter_sentence_similarity = sum_of_similarities / num_comparisons
        return inter_sentence_similarity

    def create_entity_grid(self, sentences):
        entities = defaultdict(lambda: ['_'] * len(sentences))
        
        for i, sentence in enumerate(sentences):
            for entity, role in self.extract_entities(sentence):
                entities[entity][i] = role
        
        return entities

    def extract_entities(self, sentence):
        # This is a simplified version. In practice, you'd use NLP tools
        # to extract entities and their syntactic roles (S, O, X)
        entities = []
        for word in sentence.split():
            if word.istitle():
                entities.append((word, 'S'))  # Assume all capitalized words are subjects
        return entities

    def compute_transitions(self, grid):
        transitions = defaultdict(int)
        for entity_mentions in grid.values():
            for i in range(len(entity_mentions) - 1):
                transition = (entity_mentions[i], entity_mentions[i+1])
                transitions[transition] += 1
        return transitions

    def discourse_coherence(self, sentences):
        grid = self.create_entity_grid(sentences)
        transitions = self.compute_transitions(grid)
        
        # Calculate probabilities of transitions
        total_transitions = sum(transitions.values())
        probabilities = {t: count / total_transitions for t, count in transitions.items()}
        
        # Calculate coherence score (e.g., using entropy)
        coherence_score = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        
        return coherence_score

    def apply_framework(self, references: list | str, hypotheses: list | str) -> dict:
        """
        Apply the framework to a text generation model.
        Returns:
            dict: The results of the evaluation.
        """
        results = {}
        # Calculate perplexity
        perplexity = self.calculate_perplexity(hypotheses)
        results['perplexity'] = perplexity
        # Calculate distinct-1
        distinct_1 = self.distinct_n(hypotheses, 1)
        results['distinct_1'] = distinct_1
        # Calculate distinct-2
        distinct_2 = self.distinct_n(hypotheses, 2)
        results['distinct_2'] = distinct_2
        # Calculate type-token ratio
        ttr = self.type_token_ratio(hypotheses)
        results['ttr'] = ttr
        # Calculate moving average TTR
        moving_average_ttr = self.moving_average_ttr(hypotheses)
        results['moving_average_ttr'] = moving_average_ttr
        # Calculate BLEU score
        bleu_score = self.bleu_score(hypotheses, references)
        results['bleu_score'] = bleu_score
        # Discourse coherence
        discourse_coherence = self.discourse_coherence(hypotheses)
        results['discourse_coherence'] = discourse_coherence
        # Inter-sentence similarity
        inter_sentence_similarity = self.inter_sentence_similarity(hypotheses)
        results['inter_sentence_similarity'] = inter_sentence_similarity
        
        return results


if __name__ == '__main__':

    framework = Framework()

    hypothesis = "Der schnelle braune Fuchs springt schnell über den faulen Hund."
    reference = "Der braune schnelle Fuchs springt über den Hund."
    
    result = framework.apply_framework(reference, hypothesis)
    print(result)
