#coding: utf-8

from .chemvae import load_encoder, load_decoder, load_property_predictor

class DeepScreening():
    def __init__(self, chemvae_params=None, deeperbind_params=None ):
        self.encoder = load_encoder(chemvae_params)
        self.decoder = load_decoder(chemvae_params)
