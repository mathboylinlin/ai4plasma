"""Base model classes for neural network training and inference in AI4Plasma.

This module provides a unified framework for neural network training, validation,
and inference in the AI4Plasma project. It implements the core training pipeline
infrastructure with support for distributed computing, checkpoint management, and
TensorBoard monitoring.

Model Classes
-------------
- `BaseModel`: Abstract base class for all AI4Plasma models.
- `CfgBaseModel`: Configuration-driven model wrapper.

"""

from abc import ABC, abstractmethod
import os, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.utils.io import read_json, img2gif
from ai4plasma.config import DEVICE


class BaseModel(ABC):
    """Base Model class that provides a framework for training and predicting.
    
    This class provides a unified framework for training and inference using a neural
    network as the underlying computational model. It handles model initialization,
    loss computation, and prediction with device-agnostic computation (CPU/GPU).
    
    This class is abstract and should be subclassed by concrete implementations
    that override the `train()` and `calc_loss()` methods.
    
    Attributes
    ----------
    device_id : int
        The device ID for GPU computation.
    network : torch.nn.Module
        The neural network model moved to the configured device.
    loss_func : callable, optional
        The loss function used during training.
    optimizer : torch.optim.Optimizer, optional
        The optimizer used to update network parameters.
    lr_scheduler : torch.optim.lr_scheduler, optional
        The learning rate scheduler for adaptive learning rates.
    """

    def __init__(self, network) -> None:
        """Initialize the BaseModel with a neural network.
        
        Parameters
        ----------
        network : torch.nn.Module
            The neural network model to be used for computation.
        """
        self.device_id = DEVICE.device_id
        self.network = network.to(DEVICE())
        self.loss_func = None
        self.optimizer = None
        self.lr_scheduler = None


    def prepare_train_data(self):
        """Set training data.
        
        This method should be overridden by subclasses to load and prepare
        the training dataset.
        """
        pass

    def prepare_test_data(self):
        """Set testing data.
        
        This method should be overridden by subclasses to load and prepare
        the testing dataset.
        """
        pass


    def set_loss_func(self, loss_func) -> None:
        """Set the loss function.
        
        Parameters
        ----------
        loss_func : callable
            The loss function to be used during training. Typically a function
            from torch.nn.functional or torch.nn that computes a scalar loss value.
        """
        self.loss_func = loss_func


    def set_optimizer(self, optimizer) -> None:
        """Set the optimizer.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to be used during training. Examples include Adam,
            SGD, or other PyTorch optimizers.
        """
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler) -> None:
        """Set the learning rate scheduler.
        
        Parameters
        ----------
        lr_scheduler : torch.optim.lr_scheduler
            The learning rate scheduler to be used during training for adaptive
            learning rate adjustment.
        """
        self.lr_scheduler = lr_scheduler


    @abstractmethod
    def calc_loss(self):
        """Calculate the loss.
        
        Computes the loss value for the current batch. This method should be
        overridden by subclasses to implement custom loss computation logic.
        
        Notes
        -----
        Abstract method that must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def train(self):
        """Execute one training step.
        
        Performs a single training iteration including forward pass, loss
        calculation, and backward pass with optimizer step. This method should
        be overridden by subclasses to implement custom training logic.
        
        Notes
        -----
        Abstract method that must be implemented by subclasses.
        """
        pass


    def predict(self, X):
        """Perform inference using the model.
        
        Executes the network in evaluation mode without computing gradients,
        suitable for inference on new data.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data for inference.
        
        Returns
        -------
        torch.Tensor
            Output predictions from the neural network.
        """
        self.network.eval()
        with torch.no_grad():
            return self.network(X)


class CfgBaseModel(BaseModel):
    """Configuration-driven base model with training pipeline support.
    
    Extends BaseModel with configuration file loading, checkpoint management,
    TensorBoard logging, and automated training pipeline with visualization.
    This class implements the full training lifecycle including initialization,
    per-epoch callbacks, and post-training actions.
    
    Attributes
    ----------
    cfg : dict
        Configuration dictionary loaded from JSON file.
    saved_model : dict, optional
        Loaded model checkpoint containing model state, optimizer state, and epoch.
    loss_list : list
        List of (epoch, loss_value) tuples for tracking training history.
    fig_file_list : list
        List of figure file paths for GIF animation generation.
    writer : SummaryWriter, optional
        TensorBoard summary writer for logging metrics.
    epoch : int
        Current training epoch.
    total_epochs : int
        Total number of epochs to train.
    last_epoch : int
        Last completed epoch (used for resuming training).
    """

    def __init__(self, cfg_file, network) -> None:
        """Initialize the CfgBaseModel with a neural network and configuration.
        
        Parameters
        ----------
        cfg_file : str
            Path to the JSON configuration file containing training parameters.
        network : torch.nn.Module
            The neural network model to be used for computation.
        """
        super().__init__(network)

        self.cfg = read_json(cfg_file)

        ## set initial parameters
        self.get_init_args()

        ## set parameters from JSON configuration
        self.get_json_args(self.cfg)

    def load_model_from_file(self, model_file, map_location=None):
        """Load a model checkpoint from file.
        
        Parameters
        ----------
        model_file : str
            Path to the saved model checkpoint file.
        map_location : str, optional
            Device to map the model to. If None, uses the configured default device.
        
        Returns
        -------
        dict
            Loaded checkpoint dictionary containing model, optimizer, and epoch info.
        """
        if map_location is None:
            self.saved_model = torch.load(model_file, map_location=DEVICE())
        else:
            self.saved_model = torch.load(model_file, map_location=map_location)

        return self.saved_model

    def load_weights(self):
        """Load model weights from the saved checkpoint.
        
        Loads network weights from the saved model dictionary if available.
        """
        if self.saved_model is not None:
            if 'model' in self.saved_model:
                self.network.load_state_dict(self.saved_model['model'])

    def load_optimizer(self):
        """Load optimizer state from the saved checkpoint.
        
        Loads optimizer state including momentum, accumulated gradients, etc.
        from the saved checkpoint if available.
        """
        if self.saved_model is not None:
            if 'optimizer' in self.saved_model:
                self.optimizer.load_state_dict(self.saved_model['optimizer'])

    def load_last_epoch(self):
        """Load the last completed epoch from checkpoint and compute total epochs.
        
        Updates total_epochs based on the loaded last_epoch to support resuming
        training from where it was paused.
        """
        if self.saved_model is not None:
            if 'epoch' in self.saved_model:
                self.last_epoch = self.saved_model['epoch']

        self.total_epochs = self.last_epoch + self.num_epochs

    def load_model(self, model_file, map_location=None):
        """Load a complete checkpoint including model, optimizer, and epoch.
        
        Orchestrates loading of model weights, optimizer state, and epoch info
        from a saved checkpoint file for resuming training.
        
        Parameters
        ----------
        model_file : str
            Path to the model checkpoint file.
        map_location : str, optional
            Device to map the model to. If None, uses the configured default device.
        """
        self.load_model_from_file(model_file, map_location)
        self.load_weights()
        self.load_optimizer()
        self.load_last_epoch()

    def get_init_args(self):
        """Initialize default training parameters.
        
        Sets up default values for loss function (MSE), optimizer (Adam),
        epoch tracking, and storage lists for history and figures.
        """
        self.saved_model = None
        self.loss_func = F.mse_loss  # default loss function
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)  # default optimizer
        self.last_epoch = 0  # default number of last epoch
        self.epoch = 0
        self.total_epochs = 0
        self.load_weigths_from_file = True

        ## list for saving fig names (for generating gif file)
        self.fig_file_list = []

        ## list for saving loss value
        self.loss_list = []

        ## flag to indicate actions after training
        self.flag_do_after_training = True

    def get_json_args(self, cfg):
        """Extract and configure training parameters from configuration dictionary.
        
        Parameters
        ----------
        cfg : dict
            Configuration dictionary containing training parameters such as
            learning rate, number of epochs, logging frequency, file paths, etc.
        """
        self.lr = cfg['train']['lr']
        self.num_epochs = cfg['train']['num_epochs']

        self.log_freq = cfg['train']['log_freq']
        self.log_tag = cfg['train']['log_tag']
        self.tensorboard_writer = cfg['train']['tensorboard_writer']

        self.save_freq = cfg['train']['save_freq']
        self.save_model_file = cfg['train']['save_model_file']
        self.save_net_file = cfg['train']['save_net_file']

        self.load_model_file = cfg['train']['load_model_file']
        self.save_loss_file = cfg['train']['save_loss_file']

        self.plot_gif_freq = cfg['train']['plot_gif_freq']
        self.save_gif_file = cfg['train']['save_gif_file']
        self.save_gif_tmp_path = cfg['train']['save_gif_tmp_path']
        self.remove_gif_tmp_files = cfg['train']['remove_gif_tmp_files']
        self.save_gif_duration = cfg['train']['save_gif_duration']

        ## create necessary directory
        os.makedirs(self.save_gif_tmp_path, exist_ok=True)

        ## create tensorboard writer
        if self.tensorboard_writer != "":
            self.writer = SummaryWriter(self.tensorboard_writer)
            print('%s: Tensorboard Writer Opened' % (self.log_tag))

    def get_kwargs(self, **kwargs):
        """Extract and set training kwargs (callbacks, optimizer, loss functions).
        
        Parameters
        ----------
        kwargs : dict

            - optimizer : torch.optim.Optimizer, optional
            - calc_l2_err : callable, optional
                Function to calculate L2 error for logging.
            - plot_func_training : callable, optional
                Function to generate training plots for TensorBoard.
            - plot_func_gif : callable, optional
                Function to generate plots for GIF animation.
        """
        self.optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else self.optimizer
        self.calc_l2_err = kwargs['calc_l2_err'] if 'calc_l2_err' in kwargs else None
        self.plot_func_training = kwargs['plot_func_training'] if 'plot_func_training' in kwargs else None
        self.plot_func_gif = kwargs['plot_func_gif'] if 'plot_func_gif' in kwargs else None

    def do_before_training(self, **kwargs):
        """Initialize training pipeline before starting epochs.
        
        Extracts runtime parameters from kwargs and loads pretrained model
        checkpoint if specified in configuration.
        
        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to get_kwargs() for configuration.
        """
        self.get_kwargs(**kwargs)
        self.load_model(self.load_model_file)


    def do_after_each_epoch(self):
        """Execute callbacks after each training epoch.
        
        Handles logging to TensorBoard, generating visualization plots,
        saving checkpoints, and collecting loss history based on configured
        frequencies and callbacks.
        
        Notes
        -----
        This method should be called at the end of each epoch in the training loop.
        Automatically manages checkpoint saving, TensorBoard logging, and GIF
        figure collection based on configured frequencies.
        """
        ## write to tensorboard
        if self.epoch % self.log_freq == 0:
            loss_val = self.loss.item()
            self.loss_list.append([self.epoch, loss_val])
            print('%s epoch: [%d/%d] Loss: %g' % (self.log_tag, self.epoch, self.total_epochs, loss_val))

            if self.calc_l2_err is not None:
                l2_err = self.calc_l2_err()
                print('%s epoch: [%d/%d] L2 err: %g' % (self.log_tag, self.epoch, self.total_epochs, l2_err))

            if self.tensorboard_writer != "":
                self.writer.add_scalar('Loss-%s' % (self.log_tag), loss_val, self.epoch)
                if self.calc_l2_err is not None:
                    self.writer.add_scalar('L2-err-%s' % (self.log_tag), l2_err, self.epoch)
                    
                if self.plot_func_training is not None:
                    fig = self.plot_func_training()  
                    self.writer.add_figure('Plot-%s' % (self.log_tag), figure=fig)

                ## flush
                self.writer.flush()

        ## generate gif file
        if self.epoch % self.plot_gif_freq == 0:
            if self.plot_func_gif is not None:
                fig = self.plot_func_gif()
                fig_file = '%s/%d.png' % (self.save_gif_tmp_path, self.epoch)                     
                fig.savefig(fig_file)
                self.fig_file_list.append(fig_file)

        ## save model & network
        if self.epoch % self.save_freq == 0:
            # save model with optimizer and last epoch
            torch.save({'model': self.network.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': self.epoch}, self.save_model_file)
            print('%s: %s Saved' % (self.log_tag, self.save_model_file))
        
            # save network with weights
            torch.save(self.network, self.save_net_file)
            print('%s: %s Saved' % (self.log_tag, self.save_net_file))

    def do_after_training(self):
        """Finalize training and generate outputs.
        
        Generates animated GIF from collected figures, saves loss history to disk,
        closes TensorBoard writer, and optionally removes temporary files.
        
        Notes
        -----
        Should be called after training is complete to generate final outputs
        and cleanup temporary resources.
        """
        ## save gif file
        if self.save_gif_file is not None:
            img2gif(self.fig_file_list, self.save_gif_file, self.save_gif_duration)
            ## remove tmp directory
            if self.remove_gif_tmp_files:
                shutil.rmtree(self.save_gif_tmp_path)

        ## save loss values
        if self.save_loss_file is not None:
            header = '%-15s%-15s' % ('Epoch', 'Loss')
            data = np.array(self.loss_list)
            np.savetxt(self.save_loss_file, data, fmt='%-15d%-15g', header=header)

        ## close tensorboard
        if self.tensorboard_writer != "":
            self.writer.close()
            print('%s: Tensorboard Writer Closed' % (self.log_tag))


