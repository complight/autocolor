import os
import odak
import torch
import itertools
from tqdm import tqdm


class laser_power_predictor_cnn(torch.nn.Module):
    """
    A network for laser power estimation in AutoColor.
    """
    def __init__(
                 self,
                 kernel_size = 1,
                 output_shape = [3, 3],
                 n_hidden = 10,
                 bias = False,
                 output_directory = './output',
                 device = torch.device('cpu')
                ):
        super(laser_power_predictor_cnn, self).__init__()
        self.device = device
        self.bias = bias
        self.n_hidden = n_hidden
        self.activation = torch.nn.Sigmoid()
        self.model = torch.nn.Sequential(
                                         odak.learn.models.double_convolution(
                                                                              3,
                                                                              mid_channels = self.n_hidden,
                                                                              output_channels = self.n_hidden,
                                                                              kernel_size = kernel_size,
                                                                              bias = self.bias, 
                                                                              activation = self.activation
                                                                             ),
                                         odak.learn.models.double_convolution(
                                                                              self.n_hidden,
                                                                              mid_channels = self.n_hidden,
                                                                              output_channels = self.n_hidden,
                                                                              kernel_size = kernel_size,
                                                                              bias = self.bias, 
                                                                              activation = self.activation
                                                                             ),
                                         odak.learn.models.double_convolution(
                                                                              self.n_hidden,
                                                                              mid_channels = self.n_hidden,
                                                                              output_channels = self.n_hidden,
                                                                              kernel_size = kernel_size,
                                                                              bias = self.bias, 
                                                                              activation = self.activation
                                                                             ),
                                         torch.nn.Upsample(size = [100, 100], mode = 'bilinear', align_corners = True),
                                         odak.learn.models.double_convolution(
                                                                              self.n_hidden,
                                                                              mid_channels = self.n_hidden,
                                                                              output_channels = self.n_hidden,
                                                                              kernel_size = kernel_size,
                                                                              bias = self.bias,
                                                                              activation = self.activation
                                                                             ),
                                         odak.learn.models.double_convolution(
                                                                              self.n_hidden,
                                                                              mid_channels = self.n_hidden,
                                                                              output_channels = self.n_hidden,
                                                                              kernel_size = kernel_size,
                                                                              bias = self.bias,
                                                                              activation = self.activation
                                                                             ),
                                         torch.nn.Upsample(size = [10, 10], mode = 'bilinear', align_corners = True),
                                         odak.learn.models.double_convolution(
                                                                              self.n_hidden,
                                                                              mid_channels = self.n_hidden,
                                                                              output_channels = self.n_hidden,
                                                                              kernel_size = kernel_size,
                                                                              bias = self.bias,
                                                                              activation = self.activation
                                                                             ),
                                         torch.nn.Upsample(size = output_shape, mode ='bilinear', align_corners = True),
                                         odak.learn.models.double_convolution(
                                                                              self.n_hidden,
                                                                              mid_channels = self.n_hidden,
                                                                              output_channels = self.n_hidden,
                                                                              kernel_size = kernel_size,
                                                                              bias = self.bias,
                                                                              activation = self.activation
                                                                             ),
                                         torch.nn.Conv2d(
                                                         self.n_hidden,
                                                         1,
                                                         kernel_size = kernel_size,
                                                         padding = kernel_size // 2,
                                                         bias = bias
                                                        ),
                                        ).to(self.device)
        self.loss_l2 = torch.nn.MSELoss(reduction = 'sum')
        self.output_directory = output_directory 
        odak.tools.check_directory(self.output_directory)
   
 
    def forward(self, x):
        """
        Internal function for forward model of laser power predictor.
        """
        y = self.model(x) 
        return y


    def evaluate(self, estimation, ground_truth, peak_value = 1.8):
        """
        Internal function to evaluate an estimation against a ground truth.
        Note that this evaluation is permutation in-variant, following a logic similar to Yu, Dong, Morten Kolbæk, Zheng-Hua Tan, and Jesper Jensen. "Permutation invariant training of deep models for speaker-independent multi-talker speech separation." In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 241-245. IEEE, 2017.
        """
        permutations = list(itertools.permutations(range(0, int(estimation.shape[2]))))
        loss_permutations = torch.zeros(len(permutations)) 
        estimation_processed = torch.abs(torch.cos(estimation))
        for order_id, order in enumerate(permutations):
            ground_truth_reordered = torch.cat(( 
                                                ground_truth[:, :, order[0]].unsqueeze(2),
                                                ground_truth[:, :, order[1]].unsqueeze(2),
                                                ground_truth[:, :, order[2]].unsqueeze(2)
                                               ),
                                               dim = 2
                                              )
            current_power_loss = self.loss_l2(estimation_processed, ground_truth_reordered)
            loss_permutations[order_id] = current_power_loss
        loss = loss_permutations.min()
        return loss


    def train(self, train_data_loader, test_data_loader, epochs, learning_rate, learning_rate_floor, save_every = 0):
        """
        Function to train the model.

        Parameters
        ----------
        train_data_loader     : torch.utils.data.Dataloader
                                Train dataset loader.
        test_data_loader      : torch.utils.data.Dataloader
                                Test dataset loader.
        epochs                : int
                                Number of epochs.
        learning_rate         : float
                                Learning rate.
        save_every            : int
                                Save at every.
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)
        t_steps = tqdm(range(epochs), leave = False, dynamic_ncols = True)
        for step in t_steps:
            for g in self.optimizer.param_groups:
                g['lr'] -= (learning_rate - learning_rate_floor) / epochs
                if g['lr'] < learning_rate_floor:
                    g['lr'] = learning_rate_floor
                lr = g['lr']
            total_loss = 0
            t_data = tqdm(train_data_loader, leave = False, dynamic_ncols = True)
            for x in t_data:
                self.optimizer.zero_grad()
                image = x[0].to(self.device)
                laser_powers = x[1].to(self.device)
                laser_powers_estimate = self.forward(image)
                loss = self.evaluate(laser_powers_estimate, laser_powers)
                loss.backward(retain_graph = True)
                self.optimizer.step()
                total_loss += loss.detach() / train_data_loader.__len__()
                description = 'Per sample train loss: {:.4f}'.format(loss.item())
                t_data.set_description(description)
            if step % save_every == 0:
                torch.no_grad()
                self.test(test_data_loader, step = step)
                odak.tools.check_directory(self.output_directory)
                weights_filename = '{}/{}'.format(self.output_directory, 'weights_{:04d}.pt'.format(step))
                self.save_weights(filename = weights_filename)
            description = 'Step: {}, Train Loss: {:.4f} Learning Rate: {:.4f}'.format(step, total_loss.item(), lr)
            t_steps.set_description(description)
        print(description)


    def test(self, test_data_loader, step = 0, filename = None):
        """
        Function to test the model.

        Parameters
        ----------
        test_data_loader      : torch.utils.data.Dataloader
                                Test dataset loader.
        step                  : int
                                Epoch number.
        filename              : str
                                If provided, the output will be saved to this filename.
        """
        total_test_loss = 0
        t_data = tqdm(test_data_loader, leave = False, dynamic_ncols = True)
        for x in t_data:
            torch.no_grad()
            image = x[0].to(self.device)
            laser_powers = x[1].to(self.device)
            laser_powers_estimate = self.forward(image)
            loss = self.evaluate(laser_powers_estimate, laser_powers)
            total_test_loss += loss.detach() / test_data_loader.__len__()
            description = 'Step: {}, Per sample test Loss {:.4f}'.format(step, loss.item())
            t_data.set_description(description)
            if not isinstance(filename, type(None)):
                torch.save(laser_powers_estimate.squeeze(0).squeeze(0), os.path.expanduser(filename))
                print('Save to {}.'.format(filename)) 
                print(laser_powers_estimate)
                return
        description = 'Total test loss: {:.4f}'.format(total_test_loss.item())
        print(description)


    def save_weights(self, filename='./weights.pt'):
        """
        Function to save the current weights of the network to a file.

        Parameters
        ----------
        filename        : str
                          Filename.
        """
        torch.save(self.state_dict(), os.path.expanduser(filename))


    def load_weights(self, filename='./weights.pt'):
        """
        Function to load weights for this network from a file.

        Parameters
        ----------
        filename        : str
                          Filename.
        """
        state_dict = torch.load(os.path.expanduser(filename), map_location = self.device)
        self.load_state_dict(state_dict)
