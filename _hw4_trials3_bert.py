import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# % matplotlib inline

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Let's see how to increase the vocabulary of Bert model and tokenizer
num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

# tokenizer.save_pretrained('test_dir')

text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

###################################################

# Define a new example sentence with multiple meanings of the word "bank"
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))

# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)
print (segments_ids)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    final_layer, _ = model(tokens_tensor, segments_tensors)

print ("Number of batches:", len(final_layer))
print ("Number of tokens:", len(final_layer[0]))
print ("Number of hidden units:", len(final_layer[0][0]))

# For the 5th token in our sentence, select its feature values from layer 5.
vec = final_layer[0][5]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
plt.show()

print('First 5 vector values for each instance of "bank".')
print('')
print("bank vault   ", str(final_layer[0][6][:5]))
print("bank robber  ", str(final_layer[0][10][:5]))
print("river bank   ", str(final_layer[0][19][:5]))

from scipy.spatial.distance import cosine
def cos_sim(u, v):
    return 1 - cosine(u, v)

print('Vector similarity [6,10] : ', cos_sim(final_layer[0][6], final_layer[0][10]))
print('Vector similarity [10,19] : ', cos_sim(final_layer[0][10], final_layer[0][19]))

