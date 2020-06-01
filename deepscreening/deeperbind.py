#coding: utf-8
import numpy as np
from keras.layers import Input, LSTM, Convolution1D, BatchNormalization
from keras.models import Model

def deeperbind(params={}, x_train_data={}, y_train_data={}):
    max_seq_len   = params.get("max_seq_len", 1000)
    num_base_type = params.get("num_base_type", 4)
    x_in = Input(shape=(None, num_base_type), name="input_sequence")

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

    return Model(x_in, x)
