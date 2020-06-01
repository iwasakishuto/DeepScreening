#coding: utf-8
import os
import json
import argparse
import numpy as np
from datetime import datetime

from keras.callbacks import ModelCheckpoint
from kerasy.utils import toBLUE, toGREEN, toCYAN, toRED

from deepscreening.chemvae import ChemVAE
from deepscreening.utils import arange_losses, arange_metrics, arange_optimizers
from deepscreening.utils import load_params
from deepscreening.utils import arange_all_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str)
    args = parser.parse_args()

    # NOTE: Change directory to where params file locates.
    params_rela_path = args.params
    dirname          = os.path.dirname(params_rela_path)
    params_file_name = os.path.basename(params_rela_path)
    os.chdir(dirname)
    print(f"{toRED('Change directory to')} {toGREEN(dirname)}")
    # Load parameter files.
    params = load_params(path=params_file_name, name="MLP")
    print(f"Load params from {toBLUE(params_file_name)}")
    print(json.dumps(params, indent=2))

    # Load x training data. (NOTE: Run in `dirname` directory.)
    (x_train_data,y_train_data),validation_data,_ = arange_all_datasets(params)
    print("x_train.shape: ")
    for layer,x_train in x_train_data.items():
        print(f"{layer}: {toCYAN(x_train.shape)}")
    print("y_train.shape: ")
    for layer,y_train in y_train_data.items():
        print(f"{layer}: {toCYAN(y_train.shape)}")

    # Build & Compile a Model
    model_name = params.get("model")
    model = {
        "ChemVAE" : ChemVAE,
    }.get(model_name)(params=params, x_train_data=x_train_data, y_train_data=y_train_data)
    print(f"Buit {model_name} model.")
    # Display model summary.
    with open("model_summary.txt", mode="w") as fp:
        model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
    model.summary()
    print(f"Mode summary is saved at {toBLUE('model_summary.txt')}")

    # Compile.
    optimizer   = arange_optimizers(params)
    loss_dict   = arange_losses(params)
    metric_dict = arange_metrics(params)
    model.compile(optimizer=optimizer, loss=loss_dict, metrics=metric_dict)

    # Create a directory for storing the results.
    weights_dirname = datetime.now().strftime("weights_%Y-%m-%d_%H:%M:%S")
    if not os.path.exists(weights_dirname):
        os.mkdir(weights_dirname)
    epochs           = params.get("epochs", 1)
    batch_size       = params.get("batch_size", 32)
    latent_space_dim = params.get("latent_space_dim")
    filename = f"{latent_space_dim}Demb-{batch_size}bs-{epochs}_" + "{epoch}.h5"
    filepath = os.path.join(weights_dirname, filename)
    print(f"Weight files will be saved at {filepath}.")

    # Training.
    modelCheckpoint = ModelCheckpoint(filepath=filepath, monitor="val_loss", save_best_only=True, period=1)
    history = model.fit(
        x_train_data=x_train_data, y_train_data=y_train_data, validation_data=validation_data,
        batch_size=batch_size, epochs=epochs, verbose=1,
        callbacks=[modelCheckpoint]
    )
    # Save the results.
    os.chdir(weights_dirname)
    print(f"Change directory to {toGREEN(weights_dirname)}.")
    np.savetxt("loss_history.txt", np.asarray(history.history["loss"]), delimiter=",")
    print(f"Loss history is saved at {toBLUE('loss_history.txt')}")
    model.save_weights(filename.format(epoch=epochs))
    print(f"Last weights is saved at {toBLUE(filename.format(epoch=epochs))}")
    model.save("model.h5")
    print(f"Model is saved at {toBLUE('model.h5')}")
