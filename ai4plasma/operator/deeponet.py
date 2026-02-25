"""DeepONet (Deep Operator Network) implementation for learning nonlinear operators.

This module provides a comprehensive framework for training and deploying DeepONet models,
which are neural networks designed to learn nonlinear operators mapping between
infinite-dimensional function spaces.

DeepONet Classes
----------------
- `DeepONet`: Core neural network architecture with automatic branch type detection
- `DeepONetDataset`: PyTorch Dataset supporting both 2D (FNN) and 4D (CNN) inputs
- `DeepONetModel`: High-level training wrapper with checkpointing and TensorBoard
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.core.model import BaseModel
from ai4plasma.config import DEVICE



class DeepONet(nn.Module):
    """Neural network architecture for learning nonlinear operators.

    DeepONet consists of two sub-networks: a branch network and a trunk network.
    The branch network processes input functions (supports both FNN and CNN architectures),
    while the trunk network processes spatial/temporal locations (typically FNN).
    The outputs of these networks are combined via inner product to produce the final
    prediction.

    Attributes
    ----------
    branch_net : torch.nn.Module
        Branch network (FNN or CNN) that processes input functions.
    trunk_net : torch.nn.Module
        Trunk network (typically FNN) that processes coordinates.
    bias_last : torch.nn.Parameter
        Learnable bias term added to final output.
    branch_is_cnn : bool
        Flag indicating if branch is CNN (4D) or FNN (2D).
    """

    def __init__(self, branch_net, trunk_net):
        """Initialize DeepONet with branch and trunk networks.

        Parameters
        ----------
        branch_net : torch.nn.Module
            Branch network that processes input functions.
            Can be FNN (2D input) or CNN (4D input).
        trunk_net : torch.nn.Module
            Trunk network for spatial/temporal locations.
        """
        super(DeepONet, self).__init__()

        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias_last = nn.Parameter(torch.zeros(1))  # Initialize bias term to zero
        
        # Auto-detect if branch is CNN based on class name
        self.branch_is_cnn = self._detect_cnn_branch()

    def _detect_cnn_branch(self):
        """Detect if branch network is CNN or FNN based on class name.
        
        Returns
        -------
        bool
            True if branch network is CNN, False if FNN.
        """
        branch_class_name = self.branch_net.__class__.__name__
        return branch_class_name.upper() == 'CNN'

    def forward(self, branch_inputs, trunk_inputs):
        """Forward pass through DeepONet.

        Automatically handles both FNN (2D input) and CNN (4D input) branch networks.
        Combines branch and trunk outputs via inner product (Einstein summation) and
        adds a learnable bias term.

        Parameters
        ----------
        branch_inputs : torch.Tensor
            Input data for branch network.
            
            - FNN: shape (batch_size, features)
            - CNN: shape (batch_size, channels, height, width)
        trunk_inputs : torch.Tensor
            Input for trunk network, shape (num_points, features).

        Returns
        -------
        torch.Tensor
            Output prediction of DeepONet, shape (batch_size, num_points).
        """
        # Process through branch network
        branch = self.branch_net(branch_inputs)
        # Process through trunk network
        trunk = self.trunk_net(trunk_inputs)

        # Combine outputs of branch and trunk networks using Einstein summation
        # branch shape: (batch_size, d) or (batch_size, output_dim)
        # trunk shape: (num_points, d) or (num_points, output_dim)
        out = torch.einsum("bi,ni->bn", branch, trunk)

        # Add the learnable bias term to the final output
        out += self.bias_last

        return out


class DeepONetDataset(Dataset):
    """PyTorch Dataset for DeepONet supporting FNN and CNN branch networks.
    
    This dataset handles the organization of branch-trunk-target triplets for DeepONet
    training. It supports flexible splitting strategies and automatically detects whether
    the branch input is for FNN (2D) or CNN (4D) networks.
    
    Attributes
    ----------
    branch_inputs : torch.Tensor
        Input data for the branch network.
    trunk_inputs : torch.Tensor
        Input data for the trunk network.
    targets : torch.Tensor
        Target output data.
    split_by_branch : bool
        Splitting strategy flag.
    is_cnn_input : bool
        Flag indicating if branch input is 4D (CNN) or 2D (FNN).
    """
    
    def __init__(self, branch_inputs, trunk_inputs, targets, split_by_branch=True):
        """Initialize the DeepONet dataset.

        Parameters
        ----------
        branch_inputs : torch.Tensor
            Input data for the branch network.
            
            - For FNN: shape (A, M) where A is number of samples, M is feature dimension
            - For CNN: shape (A, C, H, W) where A is number of samples, C is channels,
              H, W are spatial dims
        trunk_inputs : torch.Tensor
            Input data for the trunk network, with shape (B, N) where B is number of points,
            N is feature dimension.
        targets : torch.Tensor
            Target output data, with shape (A, B).
        split_by_branch : bool, optional
            If True, split the dataset by the sample indices of branch_inputs; if False,
            split the dataset by the sample indices of trunk_inputs. Default is True.
        """
        self.branch_inputs = branch_inputs
        self.trunk_inputs = trunk_inputs
        self.targets = targets
        self.split_by_branch = split_by_branch
        
        # Detect input shape: 2D for FNN, 4D for CNN
        self.is_cnn_input = (branch_inputs.dim() == 4)

    def __len__(self):
        """Return the size of the dataset.

        Returns
        -------
        int
            The size of the dataset (number of samples based on split strategy).
        """
        if self.split_by_branch:
            return len(self.branch_inputs)  # Split by the number of samples in branch_inputs
        else:
            return len(self.trunk_inputs)   # Split by the number of samples in trunk_inputs

    def __getitem__(self, idx):
        """Return a sample by index.
        
        Handles both 2D (FNN) and 4D (CNN) branch inputs, squeezing single dimensions
        and maintaining proper batch dimensions.

        Parameters
        ----------
        idx : int
            The index of the sample.

        Returns
        -------
        tuple
            A tuple containing (branch_input, trunk_input, target).
            
            - branch_input: For FNN: shape (M,), For CNN: shape (C, H, W)
            - trunk_input: shape (B, N) or (N,) depending on split_by_branch
            - target: shape (B,) or (A,) depending on split_by_branch
        """
        if self.split_by_branch:
            # Split by the sample indices of branch_inputs
            if self.is_cnn_input:
                # For CNN: return 3D tensor (C, H, W) instead of 4D (1, C, H, W)
                return self.branch_inputs[idx], self.trunk_inputs, self.targets[idx, :]
            else:
                # For FNN: return 1D tensor (M,) instead of 2D (1, M)
                return self.branch_inputs[idx, :], self.trunk_inputs, self.targets[idx, :]
        else:
            # Split by the sample indices of trunk_inputs
            return self.branch_inputs, self.trunk_inputs[idx, :], self.targets[:, idx]


class DeepONetModel(BaseModel):
    """High-level training and inference wrapper for DeepONet.

    This class provides a comprehensive interface for preparing training data, calculating loss,
    making predictions, and training the DeepONet model with advanced features including
    checkpointing, TensorBoard logging, learning rate scheduling, and resume capabilities.

    Attributes
    ----------
    network : torch.nn.Module
        The DeepONet model to be trained.
    writer : torch.utils.tensorboard.SummaryWriter, optional
        TensorBoard writer for logging.
    checkpoint_dir : str, optional
        Directory for saving training checkpoints.
    start_epoch : int
        Starting epoch for training (useful for resume).
    is_cnn_input : bool
        Flag indicating if branch input is CNN (4D) or FNN (2D).
    dataset : DeepONetDataset
        Dataset instance for training.
    dataloader : torch.utils.data.DataLoader
        DataLoader instance for batched training.
    """

    def __init__(self, network) -> None:
        """Initialize the DeepONetModel.

        Parameters
        ----------
        network : torch.nn.Module
            The DeepONet model to be trained (must be an instance of DeepONet).
        """
        super().__init__(network=network)
        # defaults for optional features
        self.writer = None
        self.checkpoint_dir = None
        self.start_epoch = 0


    def prepare_train_data(self, branch_input_data, trunk_input_data, target_data, split_by_branch=True, 
                           batch_size=None, shuffle=False, drop_last=False):
        """Prepare the training data and create a DataLoader.
        
        Supports both FNN and CNN branch networks. Automatically moves data to the configured
        device (CPU/GPU) and creates a custom collate function to handle mixed tensor dimensions.

        Parameters
        ----------
        branch_input_data : torch.Tensor
            Input data for the branch network.
            
            - For FNN: shape (A, M) where A is samples, M is features
            - For CNN: shape (A, C, H, W) where C is channels, H, W are spatial dims
        trunk_input_data : torch.Tensor
            Input data for the trunk network, with shape (B, N).
        target_data : torch.Tensor
            Target output data, with shape (A, B).
        split_by_branch : bool, optional
            If True, split by branch sample indices; if False, by trunk sample indices.
            Default is True.
        batch_size : int, optional
            The batch size for the DataLoader. If None, use the full dataset as a single batch.
            Default is None.
        shuffle : bool, optional
            Whether to shuffle the data in the DataLoader. Default is False.
        drop_last : bool, optional
            Whether to drop the last incomplete batch if dataset size is not divisible by
            batch_size. Default is False.
        """
        # Move data to the specified device (e.g., GPU)
        branch_input_data = branch_input_data.to(DEVICE())
        trunk_input_data = trunk_input_data.to(DEVICE())
        target_data = target_data.to(DEVICE())

        # Detect if branch input is CNN (4D) or FNN (2D)
        is_cnn_input = (branch_input_data.dim() == 4)
        self.is_cnn_input = is_cnn_input

        # Create the dataset
        self.dataset = DeepONetDataset(branch_input_data, trunk_input_data, target_data, split_by_branch)

        # Set batch size to the full dataset size if not specified
        batch_size = len(self.dataset) if batch_size is None else batch_size

        def custom_collate_fn(batch):
            """Custom collate function handling FNN (2D) and CNN (4D) branch inputs.
            
            Ensures trunk_inputs remains (B, N) and reconstructs proper batch shapes
            by stacking individual samples.

            Parameters
            ----------
            batch : list
                List of tuples returned by __getitem__.

            Returns
            -------
            tuple
                (branch_inputs, trunk_inputs, targets) with proper shapes.
                
                - FNN: branch_inputs shape (batch_size, features)
                - CNN: branch_inputs shape (batch_size, channels, height, width)
            """
            branch_inputs = [item[0] for item in batch]
            trunk_inputs = batch[0][1]  # trunk_inputs is the same for all items
            targets = [item[2] for item in batch]

            # Stack branch inputs - automatically handles both 3D (CNN) and 1D (FNN) cases
            branch_inputs = torch.stack(branch_inputs)
            targets = torch.stack(targets)

            return branch_inputs, trunk_inputs, targets

        # Create the DataLoader with custom collate_fn
        self.dataloader = DataLoader(dataset=self.dataset, 
                                     shuffle=shuffle, 
                                     batch_size=batch_size,
                                     drop_last=drop_last,
                                     collate_fn=custom_collate_fn)


    def calc_loss(self, data):
        """Calculate the training loss for a given batch.

        Parameters
        ----------
        data : tuple
            A tuple containing (branch_input, trunk_input, target).

        Returns
        -------
        torch.Tensor
            The computed loss value (scalar).
        """
        branch_input_data, trunk_input_data, target_data = data
        predict_data = self.network(branch_input_data, trunk_input_data)
        loss = self.loss_func(predict_data, target_data)

        return loss


    def predict(self, branch_input_data, trunk_input_data):
        """Perform inference using the trained DeepONet model.

        Sets the model to evaluation mode and performs forward pass without
        computing gradients.

        Parameters
        ----------
        branch_input_data : torch.Tensor
            Input data for the branch network.
        trunk_input_data : torch.Tensor
            Input data for the trunk network.

        Returns
        -------
        torch.Tensor
            The predicted output of the DeepONet model.
        """
        self.network.eval()  # Set the model to evaluation mode

        return self.network(branch_input_data, trunk_input_data)
    

    def train(self, num_epochs,
              optimizer=None,
              optimizer_cfg: dict = None,
              lr: float = 1e-4,
              lr_scheduler=None,
              print_loss: bool = True,
              print_loss_freq: int = 1,
              tensorboard_logdir: str = None,
              save_final_model: bool = False,
              final_model_path: str = None,
              checkpoint_dir: str = None,
              checkpoint_freq: int = 1,
              resume_from: str = None):
        """Train the DeepONet model with comprehensive options.

        Supports pre-built optimizer instances or configuration dictionaries for construction.
        Also supports learning-rate adjustment, TensorBoard logging, checkpointing, and resuming.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train (total, not additional when resuming).
        optimizer : torch.optim.Optimizer, optional
            Pre-built optimizer instance. If provided, used as-is. Default is None.
        optimizer_cfg : dict, optional
            Configuration to build optimizer when ``optimizer`` is None.
            Example: ``{'name': 'Adam', 'params': {'lr': 1e-4, 'weight_decay': 0}}``
            Default is None.
        lr : float, optional
            Default learning rate when building a default optimizer. Default is 1e-4.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler or callable, optional
            Optional scheduler to step each epoch. Default is None.
        print_loss : bool, optional
            Whether to print loss each epoch. Default is True.
        print_loss_freq : int, optional
            Print loss every N epochs. Default is 1.
        tensorboard_logdir : str, optional
            Path to write TensorBoard logs. If provided, a SummaryWriter is created.
            Default is None.
        save_final_model : bool, optional
            Whether to save final model at the end of training. Default is False.
        final_model_path : str, optional
            Path to save final model if save_final_model=True. Default is None.
        checkpoint_dir : str, optional
            Directory to save epoch checkpoints (model+optimizer+epoch). Default is None.
        checkpoint_freq : int, optional
            Save checkpoint every N epochs. Default is 1.
        resume_from : str, optional
            Path to checkpoint file to resume from. If provided, loads model/optimizer/epoch.
            Default is None.

        Raises
        ------
        RuntimeError
            If prepare_train_data() has not been called before training.
        FileNotFoundError
            If resume_from path does not exist.
        ValueError
            If optimizer_cfg contains an unknown optimizer name.
        """
        # Ensure dataloader exists
        if not hasattr(self, 'dataloader') or self.dataloader is None:
            raise RuntimeError('Call prepare_train_data(...) before train(...)')

        # Set default loss if not set
        if not hasattr(self, 'loss_func') or self.loss_func is None:
            self.set_loss_func(nn.MSELoss())

        # Tensorboard writer
        if tensorboard_logdir is not None and tensorboard_logdir != "":
            os.makedirs(tensorboard_logdir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_logdir)
            print(f'Tensorboard writer created at: {tensorboard_logdir}')

        # Build or use provided optimizer
        if optimizer is not None:
            self.set_optimizer(optimizer)
        else:
            # If optimizer_cfg provided, use it
            if optimizer_cfg is not None and isinstance(optimizer_cfg, dict):
                name = optimizer_cfg.get('name', 'Adam')
                params = optimizer_cfg.get('params', {})
                # if lr provided explicitly, allow override
                if 'lr' not in params and lr is not None:
                    params['lr'] = lr

                opt_cls = getattr(torch.optim, name, None)
                if opt_cls is None:
                    raise ValueError(f'Unknown optimizer name: {name}')
                self.set_optimizer(opt_cls(self.network.parameters(), **params))
            else:
                # default optimizer
                self.set_optimizer(torch.optim.Adam(self.network.parameters(), lr=lr))

        # checkpoint directory
        if checkpoint_dir is not None and checkpoint_dir != "":
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir

        # resume if requested
        self.start_epoch = 0
        if resume_from is not None and resume_from != "":
            if os.path.exists(resume_from):
                ckpt = torch.load(resume_from, map_location=DEVICE())
                if 'model' in ckpt:
                    self.network.load_state_dict(ckpt['model'])
                if 'optimizer' in ckpt and self.optimizer is not None:
                    try:
                        self.optimizer.load_state_dict(ckpt['optimizer'])
                    except Exception:
                        print('Failed to load optimizer state (incompatible). Continuing with fresh optimizer state')
                if 'epoch' in ckpt:
                    self.start_epoch = ckpt['epoch'] + 1
                print(f'Resuming from checkpoint {resume_from} at epoch {self.start_epoch}')
            else:
                raise FileNotFoundError(f'Resume checkpoint not found: {resume_from}')

        # Training loop
        total_epochs = self.start_epoch + num_epochs
        for epoch in range(self.start_epoch, total_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for branch_batch, trunk_batch, target_batch in self.dataloader:
                # Compute loss
                loss = self.calc_loss((branch_batch, trunk_batch, target_batch))

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            # learning rate scheduler step if provided
            if lr_scheduler is not None:
                # allow callable or torch scheduler
                try:
                    lr_scheduler.step()
                except Exception:
                    # maybe callable that expects epoch
                    lr_scheduler(epoch)

            avg_loss = epoch_loss / batch_count if batch_count > 0 else float('nan')

            # logging
            if print_loss and print_loss_freq > 0 and (epoch+1) % print_loss_freq == 0:
                print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:g}')

            if self.writer is not None:
                self.writer.add_scalar('Loss', avg_loss, epoch+1)
                self.writer.flush()

            # checkpointing
            if self.checkpoint_dir is not None and checkpoint_freq > 0 and ((epoch+1) % checkpoint_freq == 0):
                ckpt_file = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({'model': self.network.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': epoch}, ckpt_file)
                print(f'Checkpoint saved: {ckpt_file}')

        # final save
        if save_final_model:
            if final_model_path is None or final_model_path == "":
                if self.checkpoint_dir is not None:
                    final_model_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
                else:
                    final_model_path = 'final_model.pth'

            torch.save({'model': self.network.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': total_epochs - 1}, final_model_path)
            print(f'Final model saved to: {final_model_path}')

        # close writer
        if self.writer is not None:
            self.writer.close()
   




