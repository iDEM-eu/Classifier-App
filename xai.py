import torch
from captum.attr import LayerIntegratedGradients
import pandas as pd
from collections import defaultdict
import string
from utils import detect_language, STOPWORDS_DICT, PUNCTUATION
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Captum XAI Class
class XAI:
    def __init__(self, text, label, tokenizer, model, device):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.input_ids = None
        self.ref_input_ids = None

    def construct_input_ref(self):
        text_ids = self.tokenizer.encode(self.text, add_special_tokens=False)
        input_ids = [self.tokenizer.cls_token_id] + text_ids + [self.tokenizer.sep_token_id]
        ref_input_ids = [self.tokenizer.cls_token_id] + [self.tokenizer.pad_token_id] * len(text_ids) + [self.tokenizer.sep_token_id]
        self.input_ids = torch.tensor([input_ids], device=self.device)
        self.ref_input_ids = torch.tensor([ref_input_ids], device=self.device)
        return self.input_ids, self.ref_input_ids

    def custom_forward(self, inputs):
        return torch.softmax(self.model(inputs)[0], dim=1)[0]

    def filter_stopwords_punctuation(self, words, attributions, text):
        """
        Filters out stop words and punctuation dynamically based on detected language.
        """
        detected_lang = detect_language(text)
        stopwords_set = STOPWORDS_DICT.get(detected_lang, STOPWORDS_DICT["english"])  # Default to English if not found

        filtered_words = []
        filtered_attributions = []
        
        for word, attribution in zip(words, attributions):
            if word.lower() not in stopwords_set and word not in PUNCTUATION:
                filtered_words.append(word)
                filtered_attributions.append(attribution)
        
        return filtered_words, filtered_attributions


    def aggregate_token_attributions(self, attributions, tokens):
        """
        Aggregates token attributions to whole words.
        """
        word_attributions = defaultdict(float)
        word_list = []
        current_word = ""
        
        for i, token in enumerate(tokens):
            if token.startswith("##"):  # Handle WordPiece subword tokenization
                current_word += token[2:]  # Append subword (without "##")
            else:
                if current_word:
                    word_list.append(current_word)  # Store previous word
                current_word = token  # Start a new word

            word_attributions[current_word] += attributions[i].item()

        if current_word:
            word_list.append(current_word)

        return word_list, [word_attributions[word] for word in word_list]

    def compute_attributions(self):
        """
        Computes word-level attributions while filtering out punctuation and stop words based on language.
        """
        self.input_ids, self.ref_input_ids = self.construct_input_ref()
        self.tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[0])
        
        lig = LayerIntegratedGradients(self.custom_forward, self.model.bert.embeddings)
        attributions, delta = lig.attribute(
            inputs=self.input_ids,
            baselines=self.ref_input_ids,
            n_steps=500,
            internal_batch_size=3,
            return_convergence_delta=True
        )

        attributions = attributions.sum(dim=-1).squeeze()
        normalized_attributions = attributions / torch.norm(attributions)

        # Convert to whole-word attributions
        words, word_attributions = self.aggregate_token_attributions(normalized_attributions, self.tokens)
        
        # Filter stop words and punctuation based on language
        filtered_words, filtered_attributions = self.filter_stopwords_punctuation(words, word_attributions, self.text)

        return filtered_words, filtered_attributions



    def predict_probabilities(self):
        outputs = self.custom_forward(self.input_ids)
        probabilities = outputs.tolist()
        return probabilities

    def generate_html(self):
        words, word_attributions = self.compute_attributions()
        probabilities = self.predict_probabilities()

        token_html = ""
        for word, score in zip(words, word_attributions):
            color = f"rgba(255, 0, 0, {abs(score)})" if score < 0 else f"rgba(0, 0, 255, {abs(score)})"
            token_html += f"<span style='background-color: {color}; padding: 2px;'>{word} </span>"

        top_attributions = pd.DataFrame({
            'Word': words,
            'Attribution': word_attributions
        }).sort_values(by='Attribution', ascending=False).head(10)

        html_content = f"""
        <div style="margin-bottom: 20px;">
            <h4>Prediction Probabilities</h4>
            <div>
                <div>Simple</div>
                <div style="width: 100%; height: 20px; background-color: #ddd; border-radius: 5px; margin: 5px 0;">
                    <div style="width: {probabilities[0] * 100}%; height: 100%; background-color: blue; border-radius: 5px;"></div>
                </div>
                <p>Probability: {probabilities[0]:.2f}</p>
                <div>Complex</div>
                <div style="width: 100%; height: 20px; background-color: #ddd; border-radius: 5px; margin: 5px 0;">
                    <div style="width: {probabilities[1] * 100}%; height: 100%; background-color: orange; border-radius: 5px;"></div>
                </div>
                <p>Probability: {probabilities[1]:.2f}</p>
            </div>
            <h4>Text with Highlighted Words</h4>
            <p>{token_html}</p>
        </div>
        """
        return html_content, top_attributions

