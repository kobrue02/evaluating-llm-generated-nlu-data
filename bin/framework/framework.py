import numpy as np
import torch

from nltk import ngrams
from nltk.stem import Cistem
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from bin.utils.types import DataSet

class Framework:

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

    def calculate_coherence(self, text, model = None) -> float:
        """
        Calculate the coherence of a given text using a sentence embedding model.
        Args:
            text (str): The text to calculate the coherence of.
            model (sentence_transformers.SentenceTransformer): The sentence embedding model.
        Returns:
            float: The coherence of the text.
        """
        if not model:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = text.split('.')
        embeddings = model.encode(sentences)
        similarities = cosine_similarity(embeddings)
        coherence = np.mean(similarities)
        return coherence.astype(float)

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



if __name__ == '__main__':

    framework = Framework()

    hypothesis = "Der schnelle braune Fuchs springt schnell über den faulen Hund."
    reference = "Der braune schnelle Fuchs springt über den Hund."
    
    print(framework.bleu_score(hypothesis, reference))
    print(framework.distinct_n(hypothesis, 2))
    print(framework.calculate_coherence(hypothesis))
    print(framework.type_token_ratio(hypothesis))
    print(framework.moving_average_ttr(hypothesis))
