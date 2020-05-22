# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import (Layer, Input, Lambda, Dense, Flatten,
                                     RepeatVector, Dropout, Concatenate,
                                     Convolution1D, GRU, BatchNormalization)
from tensorflow.keras.models import load_model, Model

from .utils import load_params

class ChemVAE(Model):
    def __init__(self, x_train, param_path=None, **kwargs):
        params = load_params(path=param_path, name="chemvae")
        params.update(kwargs)

        num_tranin, max_chem_len, num_chars = x_train.shape
        params["max_chem_len"] = max_chem_len
        params["num_chars"] = num_chars

        encoder = load_encoder(params=params)
        decoder = load_decoder(params=params)
        property_predictor = load_property_predictor(params=params)

        x_in = encoder.input
        z_mean, z_log_var, z = encoder(x_in)
        reconstructed = decoder(z)
        predictions = property_predictor(z)
        super().__init__(inputs=x_in, outputs=[reconstructed, predictions])

        self.encoder = encoder
        self.decoder = decoder
        self.property_predictor = property_predictor
        self.params = params
        self.x_train = x_train

        loss = { "decoder_gru_final": params.get("reconstruction_loss", "categorical_crossentropy")}
        if "reg_prop_tasks" in params:
            loss.update({"reg_property_output": params.get("reg_prop_pred_loss", "mse")})
        if "logit_prop_tasks" in params:
            loss.update({"logit_property_output": params.get("logit_prop_pred_loss", "binary_crossentropy")})
        self.loss_dict = loss
        self.fit = {
            2: self._fit_mono_prop_task,
            3: self._fit_multi_prop_tasks,
        }.get(len(loss))

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None):
        super().compile(optimizer=optimizer, loss=self.loss_dict, metrics=metrics, loss_weights=loss_weights)

    def _fit_mono_prop_task(self, y, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_data=None, **kwargs):
        task_name = [k for k in self.loss_dict.keys() if k != "decoder_gru_final"]
        y = {"decoder_gru_final": self.x_train, task_name : y}
        if validation_data is not None:
            x_val, y_val = validation_data
            y_val = {"decoder_gru_final": x_val, task_name : y_val}
            validation_data = (x_val, y_val)
        batch_size = batch_size or self.params.get("batch_size")
        return super().fit(x=self.x_train, y=y, batch_size=batch_size,
                           epochs=epochs, verbose=verbose, callables=callbacks,
                           validation_data=validation_data, **kwargs)

    def _fit_multi_prop_tasks(self, y_reg, y_logit, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_data=None, **kwargs):
        y = {"decoder_gru_final": self.x_train, "reg_property_output": y_reg, "logit_property_output" : y_logit}
        if validation_data is not None:
            x_val, y_val_reg, y_val_logit = validation_data
            y_val = {"decoder_gru_final": x_val, "reg_property_output": y_val_reg, "logit_property_output" : y_val_logit}
            validation_data = (x_val, y_val)
            batch_size = batch_size or self.params.get("batch_size")
        return super().fit(x=self.x_train, y=y, batch_size=batch_size,
                           epochs=epochs, verbose=verbose, callables=callbacks,
                           validation_data=validation_data, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        kl_loss = -0.5*tf.reduce_mean(z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1.)
        self.add_loss(kl_loss)

# =============================
# Sampling layer
# =============================

class Sampling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_space_dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal_variable(shape=(batch_size, latent_space_dim), mean=0., scale=1.)
        z_rand = z_mean + tf.exp(0.5 * z_log_var)*epsilon
        return tf.keras.backend.in_train_phase(z_rand, z_mean)

# =============================
# Encoder
# =============================

def encoder_model(params={}, **kwargs):
    params.update(kwargs)
    max_chem_len = params.get("max_chem_len")
    num_chars    = params.get("num_chars", 35) # zinc.yml
    x_in = Input(shape=(max_chem_len, num_chars), name='input_mol_SMILES')

    # Convolutional
    num_conv_layers   = params.get("num_conv_layers", 4)
    conv_dim_depth    = params.get("conv_dim_depth", 8)
    conv_dim_width    = params.get("conv_dim_width", 8)
    conv_depth_gf     = params.get("conv_depth_gf", 1.15875438383)
    conv_width_gf     = params.get("conv_width_gf", 1.1758149644)
    conv_activation   = params.get("conv_activation", "tanh")
    conv_dropout_rate = params.get("conv_dropout_rate", 0.0)
    is_batchnorm_conv = params.get("is_batchnorm_conv", True)

    x = x_in
    for j in range(num_conv_layers):
        x = Convolution1D(filters=int(conv_dim_depth * conv_depth_gf**j),
                          kernel_size=int(conv_dim_width * conv_width_gf**j),
                          activation=conv_activation,
                          name=f"encoder_conv{j}")(x)
        if conv_dropout_rate > 0:
            x = Dropout(rate=conv_dropout_rate, name=f"encoder_conv_dropout{j}")(x)
        if is_batchnorm_conv:
            x = BatchNormalization(axis=-1, name=f"encoder_conv_norm{j}")(x)
    x = Flatten()(x)

    # Middle layers
    num_dense_layers    = params.get("num_dense_layers", 1)
    latent_space_dim    = params.get("latent space_dim", 128)
    latent_space_dim_gf = params.get("latent_space_dim_gf", 1.4928245388)
    dense_activation    = params.get("dense_activation", "tanh")
    dense_dropout_rate  = params.get("dense_dropout_rate", 0.0)
    is_batchnorm_dense  = params.get("is_batchnorm_dense", True)

    for j in range(num_dense_layers):
        x = Dense(units=int(latent_space_dim * latent_space_dim_gf**(num_dense_layers-j-1)),
                  activation=dense_activation,
                  name=f'encoder_dense{j}')(x)
        if dense_dropout_rate > 0:
            x = Dropout(rate=dense_dropout_rate, name=f"encoder_dense_dropout{j}")(x)
        if is_batchnorm_dense:
            x = BatchNormalization(axis=-1, name=f"encoder_dense_norm{j}")(x)

    z_mean    = Dense(latent_space_dim, name="latent_mean")(x)
    z_log_var = Dense(latent_space_dim, name="latent_log_var")(x)
    z = Sampling(name="encoder_output")([z_mean, z_log_var])

    return Model(x_in, [z_mean, z_log_var, z], name="encoder")

def load_encoder(params={}, **kwargs):
    if "encoder_weights_path" in params:
        path = params.get("encoder_weights_path")
        return load_model(path)
    else:
        return encoder_model(params, **kwargs)

# =============================
# Decoder
# =============================

def decoder_model(params={}, **kwargs):
    params.update(kwargs)
    max_chem_len     = params.get("max_chem_len")
    num_chars        = params.get("num_chars", 35) # zinc.yml
    latent_space_dim = params.get("latent space_dim", 128)
    z_in = Input(shape=(latent_space_dim,), name="decoder_input")

    # Middle layers
    num_dense_layers    = params.get("num_dense_layers", 1)
    latent_space_dim    = params.get("latent space_dim", 128)
    latent_space_dim_gf = params.get("latent_space_dim_gf", 1.4928245388)
    dense_activation    = params.get("dense_activation", "tanh")
    is_batchnorm_dense  = params.get("is_batchnorm_dense", True)
    dense_dropout_rate  = params.get("dense_dropout_rate", 0.0)

    z = z_in
    for j in range(num_dense_layers):
        z = Dense(units=int(latent_space_dim*latent_space_dim_gf**j),
                  activation=dense_activation,
                  name=f"decoder_dense{j}")(z)
        if dense_dropout_rate > 0:
            z = Dropout(rate=dense_dropout_rate, name=f"decoder_dense_dropout{j}")(z)
        if is_batchnorm_dense:
            z = BatchNormalization(axis=-1, name=f"decoder_dense_norm{j}")(z)

    # Necessary for using GRU vectors
    z_reps = RepeatVector(max_chem_len)(z)

    num_gru_layers   = params.get("num_gru_layers", 3)
    gru_dim          = params.get("gru_dim", 36)
    gru_activation   = params.get("gru_activation", "tanh")
    gru_dropout_rate = params.get("gru_dropout_rate", 0.0)
    is_batchnorm_gru = params.get("is_batchnorm_gru", True)

    # Encoder parts using GRUs
    x = z_reps
    if num_gru_layers > 1:
        for j in range(num_gru_layers-1):
            x_dec = GRU(units=gru_dim,
                        return_sequences=True,
                        activation=gru_activation,
                        name=f"decoder_gru{j}")(x)
            if gru_dropout_rate > 0:
                x = Dropout(rate=gru_dropout_rate, name=f"decoder_gru_dropout{j}")(x)
            if is_batchnorm_gru:
                x = BatchNormalization(axis=-1, name=f"decoder_gru_norm{j}")(x)

    x_out = GRU(units=num_chars,
                return_sequences=True,
                activation='softmax',
                name='decoder_gru_final')(x)

    return Model(z_in, x_out, name="decoder")

def load_decoder(params={}, **kwargs):
    if "decoder_weights_path" in params:
        path = params.get("decoder_weights_path")
        return load_model(path)
    else:
        return decoder_model(params, **kwargs)

# ====================
# Property Prediction
# ====================

def property_predictor_model(params={}, **kwargs):
    params.update(kwargs)

    num_prov_layers        = params.get("num_prov_layers", 3)
    latent_space_dim       = params.get("latent space_dim", 128)
    prop_hidden_dim        = params.get("prop_hidden_dim", 36)
    prop_hidden_dim_gf     = params.get("prop_hidden_dim_gf", 0.8)
    prop_pred_activation   = params.get("prop_pred_activation", "tanh")
    prop_pred_dropout_rate = params.get("prop_pred_dropout_rate", 0.0)
    is_batchnorm_prop      = params.get("is_batchnorm_prop", True)
    x_in = Input(shape=(latent_space_dim,), name='prop_pred_input')

    x = x_in
    for j in range(num_prov_layers):
        x = Dense(units=int(prop_hidden_dim * prop_hidden_dim_gf**j),
                  activation=prop_pred_activation,
                  name=f"property_predictor_dense{j}")(x)
        if prop_pred_dropout_rate > 0:
            x = Dropout(rate=prop_pred_dropout_rate, name=f"property_predictor_dropout{j}")(x)
        if is_batchnorm_prop:
            x = BatchNormalization(axis=-1, name=f"property_predictor_norm{j}")(x)

    reg_prop_tasks       = params.get("reg_prop_tasks", [])
    len_reg_prop_tasks   = len(reg_prop_tasks)
    logit_prop_tasks     = params.get("logit_prop_tasks", [])
    len_logit_prop_tasks = len(logit_prop_tasks)
    if len_reg_prop_tasks+len_logit_prop_tasks==0:
        raise ValueError("You must specify either 'regression tasks' and/or " + \
                         "'logistic tasks' for property prediction.")

    # for regression tasks
    outputs = []
    if len_reg_prop_tasks > 0:
        reg_prop_pred = Dense(units=len_reg_prop_tasks,
                              activation='linear',
                              name='reg_property_output')(x)
        outputs.append(reg_prop_pred)
    # for logistic tasks
    if len_logit_prop_tasks > 0:
        logit_prop_pred = Dense(units=len_logit_prop_tasks,
                                activation='sigmoid',
                                name='logit_property_output')(x)
        outputs.append(logit_prop_pred)
    return Model(inputs=x_in, outputs=outputs, name="property_predictor")

def load_property_predictor(params={}, **kwargs):
    if "property_pred_weights_path" in params:
        path = params.get("property_pred_weights_path")
        return load_model(path)
    else:
        return property_predictor_model(params, **kwargs)
