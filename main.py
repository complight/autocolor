import sys
import os
import argparse
import torch
import odak
from torch.utils.data import DataLoader
from model import laser_power_predictor_cnn
from utils import dataset


__title__ = 'Laser power predictor for HoloHDR'


def process(settings_filename, weights_filename = None, test_filename = None):
    settings = odak.tools.load_dictionary(settings_filename)
    odak.tools.check_directory(settings["general"]["output directory"])
    device = torch.device(settings["general"]["device"])
    model = laser_power_predictor_cnn(
                                      output_directory = settings["general"]["output directory"],
                                      n_hidden = settings["general"]["hidden layers"],
                                      kernel_size = settings["general"]["kernel size"],
                                      device = device
                                     )
    if not isinstance(weights_filename, type(None)): 
        model.load_weights(weights_filename)
        if not isinstance(test_filename, type(None)):
            test_dataset = dataset(directory = [], filename = test_filename, device = device)
            test_data_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
            filename = '{}/{}'.format(
                                      settings["general"]["output directory"],
                                      os.path.basename(test_filename).replace('png', 'pt')
                                     )
            model.test(
                       test_data_loader = test_data_loader,
                       filename = filename
                      )
            sys.exit()
    train_dataset = dataset(settings["training data"]["directory"], device = device)
    test_dataset = dataset(settings["test data"]["directory"], device = device)
    train_data_loader = DataLoader(train_dataset, batch_size = 1, shuffle = settings["training data"]["shuffle"])
    test_data_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    weights_filename = '{}/weights.pt'.format(settings["general"]["output directory"])
    try:
        model.train(
                    train_data_loader = train_data_loader, 
                    test_data_loader = test_data_loader, 
                    epochs = settings["general"]["epochs"], 
                    learning_rate = settings["general"]["learning rate"],
                    learning_rate_floor = settings["general"]["learning rate floor"],
                    save_every = settings["general"]["save every"]
                   )
        odak.tools.check_directory(settings["general"]["output directory"])
        model.save_weights(filename = weights_filename)
    except:
        odak.tools.check_directory(settings["general"]["output directory"])
        model.save_weights(filename = weights_filename)
        print('Training exited and weights are saved to {}'.format(weights_filename))


def main():
    settings_filename = './settings/sample.txt'
    input_filename = None
    weights_filename = None
    parser      = argparse.ArgumentParser(description=__title__)
    parser.add_argument(
                        '--weights',
                        type = argparse.FileType('r'),
                        help = 'Weights filename.',
                        required = False
                       )
    parser.add_argument(
                        '--input',
                        type = argparse.FileType('r'),
                        help = 'Input filename.',
                        required = False
                       )
    parser.add_argument(
                        '--settings',
                        type = argparse.FileType('r'),
                        help = 'Filename for the settings file. Default is {}'.format(settings_filename),
                        required = False
                       )
    args = parser.parse_args()
    if not isinstance(args.settings, type(None)):
        settings_filename = str(args.settings.name)
    if not isinstance(args.input, type(None)):
        input_filename = str(args.input.name)
    if not isinstance(args.weights, type(None)):
        weights_filename = str(args.weights.name)
    process(
            settings_filename = settings_filename,
            test_filename = input_filename,
            weights_filename = weights_filename
           )
    

if __name__ == '__main__':
    sys.exit(main())

