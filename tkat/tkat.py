

import os
BACKEND = 'jax'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras import random



import keras 
from keras import ops
from keras import Model, Input
from keras.layers import Layer, LSTM, Dense, Input, Add, LayerNormalization, Multiply, Reshape, Activation, TimeDistributed, Flatten, Lambda, MultiHeadAttention, Concatenate
from tkan import TKAN

@keras.utils.register_keras_serializable(name="AddAndNorm")
class AddAndNorm(Layer):
    def __init__(self, **kwargs):
        super(AddAndNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.add_layer = Add()
        self.add_layer.build(input_shape)
        self.norm_layer = LayerNormalization()
        self.norm_layer.build(self.add_layer.compute_output_shape(input_shape))
    
    def call(self, inputs):
        tmp = self.add_layer(inputs)
        tmp = self.norm_layer(tmp)
        return tmp

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Assuming all input shapes are the same

    def get_config(self):
        config = super().get_config()
        return config


@keras.utils.register_keras_serializable(name="GRN")
class Gate(Layer):
    def __init__(self, hidden_layer_size = None, **kwargs):
        super(Gate, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        

    def build(self, input_shape):
        if self.hidden_layer_size is None:
            self.hidden_layer_size = input_shape[-1]
        self.dense_layer = Dense(self.hidden_layer_size)
        self.gated_layer = Dense(self.hidden_layer_size, activation='sigmoid')
        self.dense_layer.build(input_shape)
        self.gated_layer.build(input_shape)

    def call(self, inputs):
        dense_output = self.dense_layer(inputs)
        gated_output = self.gated_layer(inputs)
        return ops.multiply(dense_output, gated_output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_layer_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_layer_size': self.hidden_layer_size,
        })
        return config


@keras.utils.register_keras_serializable(name="GRN")
class GRN(Layer):
    def __init__(self, hidden_layer_size, output_size=None, **kwargs):
        super(GRN, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

    def build(self, input_shape):
        if self.output_size is None:
            self.output_size = self.hidden_layer_size
        self.skip_layer = Dense(self.output_size)
        self.skip_layer.build(input_shape)
        
        self.hidden_layer_1 = Dense(self.hidden_layer_size, activation='elu')
        self.hidden_layer_1.build(input_shape)
        self.hidden_layer_2 = Dense(self.hidden_layer_size)
        self.hidden_layer_2.build((*input_shape[:2], self.hidden_layer_size))
        self.gate_layer = Gate(self.output_size)
        self.gate_layer.build((*input_shape[:2], self.hidden_layer_size))
        self.add_and_norm_layer = AddAndNorm()
        self.add_and_norm_layer.build([(*input_shape[:2], self.output_size),(*input_shape[:2], self.output_size)])

    def call(self, inputs):
        skip = self.skip_layer(inputs)
        hidden = self.hidden_layer_1(inputs)
        hidden = self.hidden_layer_2(hidden)
        gating_output = self.gate_layer(hidden)
        return self.add_and_norm_layer([skip, gating_output])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_layer_size': self.hidden_layer_size,
            'output_size': self.output_size,
        })
        return config


@keras.utils.register_keras_serializable(name="VariableSelectionNetwork")
class VariableSelectionNetwork(Layer):
    def __init__(self, num_hidden, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        batch_size, time_steps, embedding_dim, num_inputs = input_shape
        self.softmax = Activation('softmax')
        self.num_inputs = num_inputs
        self.flatten_dim = time_steps * embedding_dim * num_inputs
        self.reshape_layer = Reshape(target_shape=[time_steps, embedding_dim * num_inputs])
        self.reshape_layer.build(input_shape)
        self.mlp_dense = GRN(hidden_layer_size = self.num_hidden, output_size=num_inputs)
        self.mlp_dense.build((batch_size, time_steps, embedding_dim * num_inputs))
        self.grn_layers = [GRN(self.num_hidden) for _ in range(num_inputs)]
        for i in range(num_inputs):
            self.grn_layers[i].build(input_shape[:3])
        super(VariableSelectionNetwork, self).build(input_shape)

    def call(self, inputs):
        _, time_steps, embedding_dim, num_inputs = inputs.shape
        flatten = self.reshape_layer(inputs)
        # Variable selection weights
        mlp_outputs = self.mlp_dense(flatten)
        sparse_weights = ops.softmax(mlp_outputs)
        sparse_weights = ops.expand_dims(sparse_weights, axis=2)
        
        # Non-linear Processing & weight application
        trans_emb_list = []
        for i in range(num_inputs):
            grn_output = self.grn_layers[i](inputs[:, :, :, i])
            trans_emb_list.append(grn_output)
        
        transformed_embedding = ops.stack(trans_emb_list, axis=-1)
        combined = ops.multiply(sparse_weights, transformed_embedding)
        temporal_ctx = ops.sum(combined, axis=-1)
        
        return temporal_ctx

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_hidden': self.num_hidden,
        })
        return config


@keras.utils.register_keras_serializable(name="EmbeddingLayer")
class EmbeddingLayer(Layer):
    def __init__(self, num_hidden, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        self.dense_layers = [
            Dense(self.num_hidden) for _ in range(input_shape[-1])
        ]
        for i in range(input_shape[-1]):
            self.dense_layers[i].build((*input_shape[:2], 1))
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        embeddings = [dense_layer(inputs[:, :, i:i+1]) for i, dense_layer in enumerate(self.dense_layers)]
        return ops.stack(embeddings, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_hidden, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_hidden': self.num_hidden,
        })
        return config

@keras.utils.register_keras_serializable(name="RecurrentLayer")
class RecurrentLayer(Layer):
    def __init__(self, num_units, return_state=False, use_tkan=False, **kwargs):
        super(RecurrentLayer, self).__init__(**kwargs)
        self.num_units = num_units
        self.return_state = return_state
        self.use_tkan = use_tkan
        
    def build(self, input_shape):
        layer_cls = TKAN if self.use_tkan else LSTM
        self.layer = layer_cls(self.num_units, return_state=self.return_state, return_sequences=True)
        self.layer.build(input_shape)
        super(RecurrentLayer, self).build(input_shape)

    def call(self, inputs, initial_state=None):
        return self.layer(inputs, initial_state = initial_state)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_units': self.num_units,
            'return_state': self.return_state,
            'use_tkan': self.use_tkan,
        })
        return config


def TKAT(sequence_length: int, num_unknow_features: int, num_know_features: int, num_embedding: int, num_hidden: int, num_heads: int, n_ahead: int, use_tkan: bool = True):
    """Temporal Kan Transformer model

    Args:
        sequence_length (int): length of past sequence to use
        num_unknow_features (int): number of observed features (can be 0)
        num_know_features (int): number of known features (if 0 then create futures values for recurrent decoder)
        num_embedding (int): size of the embedding
        num_hidden (int): number of hidden units in layers
        num_heads (int): number of heads for multihead attention
        n_ahead (int): number of steps to predict
        use_tkan (bool, optional): Wether or not to use TKAN instead of LSTM. Defaults to True.

    Returns:
        keras.Model: The TKAT model
    """

    inputs = Input(shape=(sequence_length+n_ahead, num_unknow_features + num_know_features))

    embedded_inputs = EmbeddingLayer(num_embedding, name = 'embedding_layer')(inputs)

    past_features = Lambda(lambda x: x[:, :sequence_length, :, :], name='past_observed_and_known')(embedded_inputs)
    variable_selection_past = VariableSelectionNetwork(num_hidden, name='vsn_past_features')(past_features)

    future_features = Lambda(lambda x: x[:,sequence_length:,:,-num_know_features:], name='future_known')(embedded_inputs)
    variable_selection_future = VariableSelectionNetwork(num_hidden, name='vsn_future_features')(future_features)

    #recurrent encoder-decoder
    encode_out, *encode_states = RecurrentLayer(num_hidden, return_state = True, use_tkan = use_tkan, name='encoder')(variable_selection_past)
    decode_out = RecurrentLayer(num_hidden, return_state = False, use_tkan = use_tkan, name='decoder')(variable_selection_future, initial_state = encode_states)

    # all encoder-decod er history
    history = Concatenate(axis=1)([encode_out, decode_out])
  
    #feed forward
    selected = Concatenate(axis=1)((variable_selection_past, variable_selection_future))
    all_context = AddAndNorm()([Gate()(history), selected])

    #GRN using TKAN before attention 
    enriched = GRN(num_hidden)(all_context)

    #attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=enriched.shape[-1])(enriched, enriched, enriched)
    
    # Flatten the attention output and predict the future sequence
    flattened_output = Flatten()(attention_output)
    dense_output = Dense(n_ahead)(flattened_output)
    
    return Model(inputs=inputs, outputs=dense_output)

