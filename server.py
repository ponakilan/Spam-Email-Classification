import pickle

import torch
import streamlit as st
from torchtext.data.utils import get_tokenizer

from classifier.model import Classifier, load_model

vocab_size = 312768
hidden_size = 128
embed_size = 512
num_classes = 1
max_len = 100

tokenizer = get_tokenizer('basic_english')
tokens_to_idx = pickle.load(open('weights/tokens_to_idx.pkl', 'rb'))


def convert_to_vector(text: str) -> torch.Tensor:
    tokens = tokenizer(text)

    indices = []
    for token in tokens:
        try:
            indices.append(tokens_to_idx[token])
        except KeyError:
            indices.append(tokens_to_idx[","])

    len_tokens = len(tokens)
    if len_tokens < max_len:
        indices.extend([0 for i in range(max_len - len_tokens)])
    elif len_tokens > max_len:
        indices = indices[:max_len]

    return torch.tensor(indices)


st.title('Spam Email Classification')
text = st.text_area("Email text")

if len(text) != 0:
    vector = convert_to_vector(text)
    model = Classifier(vocab_size, hidden_size, embed_size, num_classes)
    load_model(model)
    model.eval()
    with torch.no_grad():
        outputs = model(vector.unsqueeze(0))

    if outputs[0][0] >= 0.5:
        st.text('Spam')
    else:
        st.text('Not Spam')
