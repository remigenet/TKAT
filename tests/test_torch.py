import os
BACKEND = 'torch'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras import random
from tkat import TKAT  # Assuming you've defined TKAT in a separate file

def generate_random_tensor(shape):
    return random.normal(shape=shape, dtype=backend.floatx())

def test_tkat_basic():
    assert keras.backend.backend() == BACKEND
    batch_size, sequence_length, n_ahead = 32, 10, 5
    num_unknow_features, num_know_features = 3, 2
    num_embedding, num_hidden, num_heads = 8, 16, 4

    tkat_model = TKAT(
        sequence_length=sequence_length,
        num_unknow_features=num_unknow_features,
        num_know_features=num_know_features,
        num_embedding=num_embedding,
        num_hidden=num_hidden,
        num_heads=num_heads,
        n_ahead=n_ahead
    )

    input_shape = (batch_size, sequence_length + n_ahead, num_unknow_features + num_know_features)
    input_data = generate_random_tensor(input_shape)
    output = tkat_model(input_data)

    expected_output_shape = (batch_size, n_ahead)
    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"

def test_tkat_variable_selection():
    assert keras.backend.backend() == BACKEND
    batch_size, sequence_length, n_ahead = 16, 8, 3
    num_unknow_features, num_know_features = 4, 3
    num_embedding, num_hidden, num_heads = 4, 8, 2

    tkat_model = TKAT(
        sequence_length=sequence_length,
        num_unknow_features=num_unknow_features,
        num_know_features=num_know_features,
        num_embedding=num_embedding,
        num_hidden=num_hidden,
        num_heads=num_heads,
        n_ahead=n_ahead
    )

    input_shape = (batch_size, sequence_length + n_ahead, num_unknow_features + num_know_features)
    input_data = generate_random_tensor(input_shape)
    
    # Get the embedding layer output
    embedding_layer = tkat_model.get_layer('embedding_layer')  # Assuming you've named your EmbeddingLayer
    embedded_input = embedding_layer(input_data)
    
    # Access the variable selection networks
    vsn_past = tkat_model.get_layer('vsn_past_features')
    vsn_future = tkat_model.get_layer('vsn_future_features')

    # Test VSN outputs
    past_features = embedded_input[:, :sequence_length, :, :]
    future_features = embedded_input[:, sequence_length:, :, -num_know_features:]
    
    past_output = vsn_past(past_features)
    future_output = vsn_future(future_features)

    assert past_output.shape == (batch_size, sequence_length, num_hidden)
    assert future_output.shape == (batch_size, n_ahead, num_hidden)



def test_tkat_attention():
    assert keras.backend.backend() == BACKEND
    batch_size, sequence_length, n_ahead = 8, 6, 2
    num_unknow_features, num_know_features = 4, 3
    num_embedding, num_hidden, num_heads = 4, 8, 2

    tkat_model = TKAT(
        sequence_length=sequence_length,
        num_unknow_features=num_unknow_features,
        num_know_features=num_know_features,
        num_embedding=num_embedding,
        num_hidden=num_hidden,
        num_heads=num_heads,
        n_ahead=n_ahead
    )

    input_shape = (batch_size, sequence_length + n_ahead, num_unknow_features + num_know_features)
    input_data = generate_random_tensor(input_shape)
    
    # Get the attention layer
    attention_layer = next(layer for layer in tkat_model.layers if isinstance(layer, keras.layers.MultiHeadAttention))

    # Test attention output
    output = tkat_model(input_data)
    assert output.shape == (batch_size, n_ahead)

def test_tkat_training():
    assert keras.backend.backend() == BACKEND
    batch_size, sequence_length, n_ahead = 64, 12, 4
    num_unknow_features, num_know_features = 4, 3
    num_embedding, num_hidden, num_heads = 8, 16, 4

    tkat_model = TKAT(
        sequence_length=sequence_length,
        num_unknow_features=num_unknow_features,
        num_know_features=num_know_features,
        num_embedding=num_embedding,
        num_hidden=num_hidden,
        num_heads=num_heads,
        n_ahead=n_ahead
    )

    input_shape = (batch_size, sequence_length + n_ahead, num_unknow_features + num_know_features)
    input_data = generate_random_tensor(input_shape)
    target_data = generate_random_tensor((batch_size, n_ahead))

    tkat_model.compile(optimizer='adam', loss='mse')
    history = tkat_model.fit(input_data, target_data, epochs=2, batch_size=16, verbose=0)

    assert len(history.history['loss']) == 2
    assert history.history['loss'][1] < history.history['loss'][0]

def test_tkat_prediction():
    assert keras.backend.backend() == BACKEND
    batch_size, sequence_length, n_ahead = 32, 10, 5
    num_unknow_features, num_know_features = 3, 2
    num_embedding, num_hidden, num_heads = 8, 16, 4

    tkat_model = TKAT(
        sequence_length=sequence_length,
        num_unknow_features=num_unknow_features,
        num_know_features=num_know_features,
        num_embedding=num_embedding,
        num_hidden=num_hidden,
        num_heads=num_heads,
        n_ahead=n_ahead
    )

    input_shape = (batch_size, sequence_length + n_ahead, num_unknow_features + num_know_features)
    input_data = generate_random_tensor(input_shape)

    predictions = tkat_model.predict(input_data)
    assert predictions.shape == (batch_size, n_ahead)

if __name__ == "__main__":
    pytest.main([__file__])