#coding: utf-8
import warnings
import numpy as np
from keras.layers import (Input, Masking, Convolution1D, LSTM, Dense,
                          Dropout, BatchNormalization)
from keras.models import Model

from .utils import load_params
from .utils import update_params

class DeeperBind(Model):
    def __init__(self, params=None, x_train_data={}, y_train_data={}, **kwargs):
        if params is None or isinstance(params, str):
            params = load_params(path=params, name="deeperbind")
        params.update(kwargs)
        params = self._update_params(params, x_train_data, y_train_data)

        # Build the models.
        x_in, outputs = self.build_model(params)
        super().__init__(inputs=x_in, outputs=outputs, name="DeeperBind")
        self.params = params
        warnings.warn("Since keras.layers.Convolution1D does not support keras.layers.Masking, " + \
                      "you have to pad the input array with zeros to virtually eliminate the amount of information. " + \
                      "This should be covered by Mask in the future.")

    def _update_params(self, params, x_train_data={}, y_train_data={}):
        if "input_sequence" in x_train_data:
            x_train_input = x_train_data.get("input_sequence")
            num_tranin, max_seq_len, num_base_types = x_train_input.shape
            params = update_params(params, max_seq_len=max_seq_len, num_base_types=num_base_types)
        return params

    def build_model(self, params):
        max_seq_len    = params.get("max_seq_len")
        num_base_types = params.get("num_base_types")
        if (max_seq_len is None) or (num_base_types is None):
            raise ValueError("You should define `max_seq_len` and `num_base_types` in parameter file, " + \
                             "or give the training data as an argument `x_train_data` to define them automatically.")

        # TODE:
        # Pattern.1: Convolution1D doesnot support Masking layer.
        # x_in = Input(shape=(max_seq_len, num_base_types), name="input_sequence")
        # x    = Masking(mask_value=-1, name="input_mask")(x_in)
        # Pattern.2: Numpy doesnot support arrays of different length.
        # x_in = Input(shape=(None, num_base_types), name="input_sequence")
        # *Pattern.3: Pad input array with zeros to virtually eliminate the amount of information.
        x_in = Input(shape=(max_seq_len, num_base_types), name="input_sequence")

        # Convolutional
        num_conv_layers   = params.get("num_conv_layers", 6)
        conv_dim_depth    = params.get("conv_dim_depth", 8)
        conv_dim_width    = params.get("conv_dim_width", 6)
        conv_depth_gf     = params.get("conv_depth_gf", 1.15875438383)
        conv_width_gf     = params.get("conv_width_gf", 1.1758149644)
        conv_activation   = params.get("conv_activation", "tanh")
        is_batchnorm_conv = params.get("is_batchnorm_conv", True)

        x = x_in
        for j in range(num_conv_layers):
            strides=3 if j==0 else 1
            x = Convolution1D(filters=int(conv_dim_depth * conv_depth_gf**j),
                              kernel_size=int(conv_dim_width * conv_width_gf**j),
                              strides=strides, # codon
                              activation=conv_activation,
                              name=f"deeperbind_conv{j}")(x)
            if is_batchnorm_conv:
                x = BatchNormalization(axis=-1, name=f"encoder_conv_norm{j}")(x)

        # stacked LSTM layers
        num_lstm_layers     = params.get("num_lstm_layers", 2)
        latent_space_dim    = params.get("latent_space_dim", 128)
        latent_space_dim_gf = params.get("latent_space_dim_gf", 1.4928245388)
        lstm_activation     = params.get("lstm_activation", "tanh")

        return_sequences = True
        for j in range(num_lstm_layers):
            if j==num_lstm_layers-1:
                return_sequences = False
            x = LSTM(units=int(latent_space_dim * latent_space_dim_gf**(num_lstm_layers-j-1)),
                     return_sequences=return_sequences,
                     activation=lstm_activation,
                     name=f"deeperbind_lstm{j}")(x)

        # Dense layers
        num_dense_layers   = params.get("num_dense_layers", 2)
        dense_activation   = params.get("dense_activation", "tanh")
        is_batchnorm_dense = params.get("is_batchnorm_dense", True)
        dense_dropout_rate = params.get("dense_dropout_rate", 0.25)

        for j in range(num_dense_layers-1):
            x = Dense(units=latent_space_dim,
                      activation=dense_activation,
                      name=f"deeperbind_dense{j}")(x)
            if dense_dropout_rate > 0 and j!=num_dense_layers-1:
                x = Dropout(rate=dense_dropout_rate**(j+1))(x)
            if is_batchnorm_dense:
                x = BatchNormalization(axis=-1, name=f"deeperbind_dense_norm{j}")(x)
        x = Dense(units=latent_space_dim, activation="linear", name="embedding")(x)
        return (x_in, x)
