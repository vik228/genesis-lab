"""
papers/bengio_et_al_(2003)_a_neural_probabilistic_language_model/model.py

Expectation:
Implement ONLY the Neural Network architecture for the Bengio (2003) Neural Probabilistic Language Model here. Do not include training loops or datasets.
    
    - `BengioNPLM(nn.Module)`:
        - `__init__(self, vocab_size, embed_dim, hidden_dim, context_len)`:
            1. Define `self.C` as `nn.Embedding(vocab_size, embed_dim)`.
            2. Define hidden layer `self.W1`.
            3. Define final/direct layer `self.W2`.
        - `forward(self, x)`: 
            1. Look up embeddings `C(x)`.
            2. Reshape context vectors.
            3. Apply hidden activation (tanh / W1).
            4. Add direct connections + W2.
            5. Return logits. DO NOT apply softmax here if using `nn.CrossEntropyLoss`.
"""
