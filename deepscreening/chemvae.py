# coding: utf-8
from keras.layers import Input, Lambda, Dense, Flatten, RepeatVector, Dropout, Concatenate
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model
from keras import backend as K
from .tgru_k2_gpu import TerminalGRU

# =============================
# Encoder
# =============================

def encoder_model(params={}):
    max_len   = params.get("max_len")
    num_chars = params.get("num_chars", 35) # zinc.yml
    x_in = Input(shape=(max_len, num_chars), name='input_mol_SMILES')

    # Convolutional
    num_conv_layers   = params.get("num_conv_layers", 4)
    conv_dim_depth    = params.get("conv_dim_depth", 8)
    conv_dim_width    = params.get("conv_dim_width", 8)
    conv_depth_gf     = params.get("conv_depth_gf", 1.15875438383)
    conv_width_gf     = params.get("conv_width_gf", 1.1758149644)
    conv_activation   = params.get("conv_activation", "tanh")
    is_batchnorm_conv = params.get("is_batchnorm_conv", True)

    x = x_in
    for j in range(num_conv_layers):
        x = Convolution1D(filters=int(conv_dim_depth * conv_depth_gf**j),
                          kernel_size=int(conv_dim_width * conv_w_growth_factor**j),
                          activation=conv_activation,
                          name=f"encoder_conv{j}")(x)
        if is_batchnorm_conv:
            x = BatchNormalization(axis=-1, name=f"encoder_conv_norm{j}")(x)
    x = Flatten()(x)

    # Middle layers
    num_dense_layers    = params.get("num_dense_layers", 1)
    latent_space_dim    = params.get("latent space_dim", 128)
    latent_space_dim_gf = params.get("latent_space_dim_gf", 1.4928245388)
    dense_activation    = params.get("dense_activation", "tanh")
    is_batchnorm_dense  = params.get("is_batchnorm_dense", True)
    dense_dropout_rate  = params.get("dense_dropout_rate", 0.0)

    for j in range(num_dense_layers):
        x = Dense(units=int(latent_space_dim * latent_space_dim_gf**(num_dense_layers-j-1)),
                  activation=dense_activation,
                  name=f'encoder_dense{j}')(x)
        if dense_dropout_rate > 0:
            x = Dropout(rate=dense_dropout_rate)(x)
        if is_batchnorm_dense:
            x = BatchNormalization(axis=-1, name=f"encoder_dense_norm{j}")(x)
    z_mean = Dense(latent_space_dim, name='z_mean_sample')(x)

    return Model(x_in, [z_mean, x], name="encoder")

def load_encoder(params={}):
    path = params.get("encoder_weights_path")
    return load_model(path)


# =============================
# Decoder
# =============================

def decoder_model(params={}):
    max_len   = params.get("max_len")
    num_chars = params.get("num_chars", 35) # zinc.yml
    latent_space_dim    = params.get("latent space_dim", 128)
    z_in = Input(shape=(latent_space_dim,), name="decoder_input")
    true_seq_in = Input(shape=(max_len, num_chars), name="decoder_true_SMILES_input")

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
            z = Dropout(rate=dense_dropout_rate)(z)
        if is_batchnorm_dense:
            z = BatchNormalization(axis=-1, name=f"decoder_dense_norm{j}")(z)

    # Necessary for using GRU vectors
    z_reps = RepeatVector(max_len)(z)

    num_gru_layers = params.get("num_gru_layers", 4)
    recurrent_dim = params.get("recurrent_dim", 50)
    rnn_activation = params.get("rnn_activation", "tanh")
    use_tgru = params.get("use_tgru" True)

    # Encoder parts using GRUs
    x_dec = z_reps
    if num_gru_layers > 1:
        for j in range(num_gru_layers-1):
            x_dec = GRU(units=recurrent_dim,
                        return_sequences=True,
                        activation=rnn_activation,
                        name=f"decoder_gru{j}")(x_dec)
    if use_tgru:
        rand_seed = params.get("rand_seed", 42)
        tgru_dropout_rate = params.get("tgru_dropout_rate", 0.0)
        terminal_GRU_implementation = params.get("terminal_GRU_implementation", 0)
        x_out = TerminalGRU(units=num_chars,
                            rnd_seed=rand_seed,
                            recurrent_dropout=tgru_dropout_rate,
                            return_sequences=True,
                            activation='softmax',
                            temperature=0.01,
                            name='decoder_tgru',
                            implementation=terminal_GRU_implementation)([x_dec, true_seq_in])
    else:
        x_out = GRU(units=num_chars,
                    return_sequences=True,
                    activation='softmax',
                    name='decoder_gru_final')(x_dec)

    if use_tgru:
        return Model([z_in, true_seq_in], x_out, name="decoder")
    else:
        return Model(z_in, x_out, name="decoder")

def load_decoder(params={}):
    path = params.get("decoder_weights_path")
    return load_model(path, custom_objects={'TerminalGRU': TerminalGRU})


##====================
## Middle part (var)
##====================

def variational_layers(z_mean, enc, kl_loss_var, params={}):
    """
    @params z_mean      : mean generated from encoder.
    @params enc         : output generated by encoding.
    @params kl_loss_var : Kullback-Leibler divergence loss.
    """
    batch_size       = params.get("batch_size", 32)
    latent_space_dim = params.get("latent space_dim", 128)
    is_batchnorm_vae = params.get("batchnorm_vae", False)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal_variable(shape=(batch_size, latent_space_dim),
                                           mean=0., scale=1.)
        # insert kl loss here
        z_rand = z_mean + K.exp(z_log_var / 2) * kl_loss_var * epsilon
        return K.in_train_phase(z_rand, z_mean)

    # variational encoding
    z_log_var = Dense(latent_space_dim, name='z_log_var_sample')(enc)

    z_samp = Lambda(sampling)([z_mean, z_log_var])
    z_mean_log_var_output = Concatenate(name='z_mean_log_var')([z_mean, z_log_var])
    if is_batchnorm_vae:
        z_samp = BatchNormalization(axis=-1)(z_samp)

    return z_samp, z_mean_log_var_output

# ====================
# Property Prediction
# ====================

def property_predictor_model(params={}):
    num_prov_layers      = params.get("num_prov_layers", 3)
    latent_space_dim     = params.get("latent space_dim", 128)
    prop_hidden_dim      = params.get("prop_hidden_dim", 36)
    prop_hidden_dim_gf   = params.get("prop_hidden_dim_gf", 0.8)
    prop_pred_activation = params.get("prop_pred_activation", "tanh")
    prop_pred_dropout    = params.get("prop_pred_dropout", 0.0)
    is_batchnorm_prop    = params.get("is_batchnorm_prop", True)
    ls_in = Input(shape=(latent_space_dim,), name='prop_pred_input')

    x = is_in
    for j in range(num_prov_layers):
        x = Dense(units=int(prop_hidden_dim * prop_hidden_dim_gf**j),
                  activation=prop_pred_activation,
                  name=f"property_predictor_dense{j}")(x)
        if prop_pred_dropout > 0:
            x = Dropout(rate=prop_pred_dropout)(x)
        if is_batchnorm_prop:
            x = BatchNormalization(axis=-1, name=f"property_predictor_norm{j}")(x)

    reg_prop_tasks = params.get("reg_prop_tasks", [])
    len_reg_prop_tasks = len(reg_prop_tasks)
    logit_prop_tasks = params.get("logit_prop_tasks", [])
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
    return Model(inputs=ls_in, outputs=outputs, name="property_predictor")

def load_property_predictor(params):
    path = params.get("property_pred_weights_path")
    return load_model(path)
