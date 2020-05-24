# coding: utf-8
import os
import re
import argparse
import warnings
import numpy as np
from keras.layers import (Layer, Input, Lambda, Dense, Flatten,
                                     RepeatVector, Dropout, Concatenate,
                                     Convolution1D, GRU, BatchNormalization)
from keras.models import load_model, Model
from keras import losses
from keras import backend as K

from .utils import load_params

class ChemVAE(Model):
    def __init__(self, params=None, x_train=None, **kwargs):
        if params is None or isinstance(params, str):
            params = load_params(path=params, name="chemvae")
        params.update(kwargs)
        if x_train is not None:
            params = self._update_params(x_train, params)
        # Build the respective models.
        encoder = load_encoder(params=params)
        decoder = load_decoder(params=params)
        property_predictor = load_property_predictor(params=params)
        # Integrates everything.
        x_in = encoder.input
        z_mean, z_log_var, z = encoder(x_in)
        reconstructed = decoder(z)
        predictions = property_predictor(z)

        if isinstance(predictions, list):
            outputs = [Lambda(identity, name=re.sub(r"^.*\/(.+_property_)output\/.*$", r"\1pred", pred.name))(pred) for pred in predictions]
            outputs.append(reconstructed)
        else:
            predictions = Lambda(identity, name=re.sub(r"^.*\/(.+_property_)output\/.*$", r"\1pred", predictions[0].name))(predictions)
            outputs = [predictions, reconstructed]
        super().__init__(inputs=x_in, outputs=outputs, name="ChemVAE")
        # Memorize.
        self.encoder = encoder
        self.decoder = decoder
        self.property_predictor = property_predictor
        # Add losses.
        self._add_losses(
            x_in=x_in, z_mean=z_mean, z_log_var=z_log_var,
            reconstructed=reconstructed, predictions=predictions,
            params=params
        )
        self.params = params

    def _update_params(self, x_train, params):
        num_tranin, max_chem_len, num_chars = x_train.shape
        params["max_chem_len"] = max_chem_len
        params["num_chars"] = num_chars
        return params

    def _add_losses(self, x_in, z_mean, z_log_var, reconstructed, predictions, params={}):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss /= params["max_chem_len"]*params["num_chars"]
        self.add_loss(K.mean(kl_loss))

        prop_losses = {"decoder": params.get("reconstruction_loss", "binary_crossentropy")}
        if "reg_prop_pred_loss" in params:
            prop_losses["reg_property_pred"] = params.get("reg_prop_pred_loss")
        if "logit_prop_pred_loss" in params:
            prop_losses["logit_property_pred"] =  params.get("logit_prop_pred_loss")
        self.prop_losses = prop_losses

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None):
        if loss is not None:
            warnings.warn(f"Loss is already defined. If you want to customize it, please describe it in the params file.")
        super().compile(optimizer=optimizer, loss=self.prop_losses, metrics=metrics, loss_weights=loss_weights)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False, **kwargs):
        num_prop_tasks = len(self.prop_losses)
        if num_prop_tasks == 3:
            y = {"decoder": x, "reg_property_pred": y[0], "logit_property_pred": y[1]}
            if validation_data is not None:
                x_val, y_val_reg, y_val_logit = validation_data
                validation_data = (x_val, {"decoder": x_val, "reg_property_pred": y_val_reg, "logit_property_pred": y_val_logit})
        elif num_prop_tasks == 2:
            prop_pred_name = list(self.prop_losses.keys())[0]
            y = {"decoder": x, prop_pred_name: y}
            if validation_data is not None:
                x_val, y_val  = validation_data
                validation_data = (x_val, {"decoder": x_val, prop_pred_name: y_val})
        return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                           callbacks=callbacks, validation_split=validation_split, validation_data=validation_data,
                           shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch,
                           steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq,
                           max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, **kwargs)

# =============================
# Lambda layer
# =============================

def identity(x):
    return K.identity(x)

def sampling(args):
    """
    reparameterization trick
    instead of sampling from Q(z|X), sample epsilon = N(0,I)
    z = z_mean + sqrt(var) * epsilon
    ~~~
    @params args (tensor): mean and log of variance of Q(z|X)
    @return z    (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    z_rand = z_mean + K.exp(0.5 * z_log_var)*epsilon
    return K.in_train_phase(z_rand, z_mean)

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
    z = Lambda(function=sampling, output_shape=(latent_space_dim,), name="encoder_output")([z_mean, z_log_var])

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

def decoder_model(params={}, add_loss=False, **kwargs):
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

    num_reg_prop_tasks       = params.get("num_reg_prop_tasks", 0)
    num_logit_prop_tasks     = params.get("num_logit_prop_tasks", 0)
    if num_reg_prop_tasks+num_logit_prop_tasks==0:
        raise ValueError("You must specify either 'regression tasks' and/or " + \
                         "'logistic tasks' for property prediction.")

    # for regression tasks
    outputs = []
    if num_reg_prop_tasks > 0:
        reg_prop_pred = Dense(units=num_reg_prop_tasks,
                              activation='linear',
                              name='reg_property_output')(x)
        outputs.append(reg_prop_pred)
    # for logistic tasks
    if num_logit_prop_tasks > 0:
        logit_prop_pred = Dense(units=num_logit_prop_tasks,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, required=True)
    parser.add_argument("-train", "--train-basename", type=str)
    parser.add_argument("-val",   "--validation-basename", type=str)
    parser.add_argument("--non-verbose", action="store_false")
    args = parser.parse_args()

    params_path = args.params
    params = load_params(params_path)
    dirname = os.path.dirname(params_path)

    train_basename = args.train_basename
    if train_basename is None:
        train_basename = os.path.join(dirname, params.get("train_basename"))
        train_x = np.load(f"{train_basename}_x.npy", allow_pickle=True)
        train_y = []
        for path in [f"{train_basename}_y_reg.npy", f"{train_basename}_y_logit.npy"]:
            if os.path.exists(path):
                train_y.append(np.load(path, allow_pickle=True))
    else:
        train_basename = os.path.basename(train_basename)

    validation_basename = args.validation_basename
    if validation_basename is None:
        validation_basename = params.get("validation_basename")
        if validation_basename is None:
            validation_data = None
        else:
            validation_basename = os.path.join(dirname, validation_basename)
            val_x = np.load(f"{validation_basename}_x.npy", allow_pickle=True)
            val_y = []
            for path in [f"{validation_basename}_y_reg.npy", f"{validation_basename}_y_logit.npy"]:
                if os.path.exists(path):
                    train_y.append(np.load(path, allow_pickle=True))
            validation_data = (val_x, val_y)
    else:
        validation_basename = os.path.basename(validation_basename)

    model = ChemVAE(params_path=params, x_train=train_x)
    with open(os.path.join(dirname, "model_summary.txt"), mode="w") as fp:
        model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

    optimizer  = params.get("optimizer", "adam")
    epochs     = params.get("epochs", 1)
    batch_size = params.get("batch_size", 32)
    verbose = 1 if args.non_verbose else 0
    model.compile(optimizer=optimizer)
    history = model.fit(x=train_x, y=train_y, epochs=epochs, verbose=verbose,
                        batch_size=batch_size, validation_data=validation_data)
    model.save_weights(os.path.join(dirname, "weights.h5"))
    model.save(os.path.join(dirname, "model.h5"))
    np.savetxt(os.path.join(dirname, "loss_history.txt"), np.asarray(history.history["loss"]), delimiter=",")
