"""DeepCSNet - Deep learning for electron-impact cross section prediction.

This module provides a specialized neural network architecture for predicting
electron-impact cross sections in plasma physics applications. DeepCSNet employs
a novel coefficient-subnet structure that separately processes different input
feature types before combining them through tensor operations.

DeepCSNet Classes
-----------------
- `DeepCSNet`: Modular neural network with coefficient-subnet architecture
- `DeepCSNetDataset`: Custom dataset class for cross section data handling
- `DeepCSNetModel`: High-level wrapper for training and inference

DeepCSNet References
--------------------
[1] Y. Wang and L. Zhong, "DeepCSNet: a deep learning method for predicting
    electron-impact doubly differential ionization cross sections,"
    Plasma Sources Science and Technology, vol. 33, no. 10, p. 105012, 2024.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.core.model import BaseModel
from ai4plasma.config import DEVICE



class DeepCSNet(nn.Module):
    """Coefficient-subnet neural network for cross section prediction.

    Architecture
    ------------ 
    The network consists of up to three optional sub-networks that process
    different input modalities:

    1. **Molecule Net** (optional, for multi-molecule scenarios):
       
       - Processes molecular descriptors
       - Input: [batch_size, molecule_features]
       - Output: [batch_size, molecule_hidden_dim]
    
    2. **Energy Net** (optional, for energy-dependent cross sections):
       
       - Processes incident energy features
       - Input: [batch_size, energy_features]
       - Output: [batch_size, energy_hidden_dim]
    
    3. **Trunk Net** (required):
       
       - Processes output coordinate features
       - Input: [n_points, coordinate_features]
       - Output: [n_points, trunk_hidden_dim]

    Operation Modes
    ---------------
    - **SMC (Single-Molecule)**: Energy Net + Trunk Net
    - **MMC (Multi-Molecule)**: Molecule Net + (optional Energy Net) + Trunk Net

    Attributes
    ----------
    molecule_net : torch.nn.Module or None
        Sub-network for processing molecular features.
        If None, operates in SMC mode.
    energy_net : torch.nn.Module or None
        Sub-network for processing energy features.
        If None in MMC mode, only molecule features are used.
    trunk_net : torch.nn.Module
        Sub-network for processing coordinate features (output dimensions).
        Required.
    bias_last : torch.nn.Parameter
        Learnable scalar bias term added to final output. Shape: [1].
    module : str
        Operating mode: "SMC" or "MMC".
    
    Notes
    -----
    - At least one of molecule_net or energy_net must be provided
    - Output dimensions of all branch networks must match trunk network output dimension
    - In MMC mode with both networks, outputs are concatenated
    """

    def __init__(self,
                 trunk_net: nn.Module,
                 molecule_net: nn.Module = None,
                 energy_net: nn.Module = None,
                 ):
        """Initialize the DeepCSNet model.
        
        Parameters
        ----------
        trunk_net : torch.nn.Module
            Sub-network for processing coordinate features (output dimensions).
            Input shape: [n_points, coordinate_dim]
            Output shape: [n_points, trunk_hidden_dim]
            Required and processes output coordinate space.
        molecule_net : torch.nn.Module, optional
            Sub-network for processing molecular features.
            Input shape: [batch_size, molecule_dim]
            Output shape: [batch_size, molecule_hidden_dim]
            If provided, operates in MMC mode. Default is None.
        energy_net : torch.nn.Module, optional
            Sub-network for processing energy features.
            Input shape: [batch_size, energy_dim]
            Output shape: [batch_size, energy_hidden_dim]
            Required in SMC mode, optional in MMC mode. Default is None.
        
        Raises
        ------
        ValueError
            If trunk_net is None (trunk network is mandatory).
            If both molecule_net and energy_net are None (at least one required).
        """
        super(DeepCSNet, self).__init__()

        if trunk_net is None:
            raise ValueError("Trunk net must be provided for DeepCSNet.")
        if molecule_net is None and energy_net is None:
            raise ValueError("At least one of molecule_net or energy_net must be provided for DeepCSNet.")

        self.molecule_net = molecule_net
        self.energy_net = energy_net
        self.trunk_net = trunk_net

        self.bias_last = nn.Parameter(torch.zeros(1))   # Initialize bis term to zero

        self.module = "SMC" if molecule_net is None else "MMC"  # Determine the module type based on the presence of molecule_net

    def forward(self, trunk_input, molecule_input=None, energy_input=None):
        """Forward pass through DeepCSNet.
        
        Processes inputs through respective sub-networks and combines outputs
        via tensor product operations.
        
        Parameters
        ----------
        trunk_input : torch.Tensor
            Input for trunk network (coordinate features).
            Shape: [n_points, coordinate_dim]
            Examples: scattering angles, ejected electron energies.
        molecule_input : torch.Tensor, optional
            Input for molecule network (molecular features).
            Shape: [batch_size, molecule_dim]
            Required in MMC mode, should be None in SMC mode.
        energy_input : torch.Tensor, optional
            Input for energy network (energy features).
            Shape: [batch_size, energy_dim]
            Required in SMC mode, optional in MMC mode.
        
        Returns
        -------
        torch.Tensor
            Predicted cross section values.
            Shape: [batch_size, n_points]
            Each row represents cross sections at all output coordinates.
        """
        trunk_output = self.trunk_net(trunk_input)

        if self.module == "SMC":
            energy_output = self.energy_net(energy_input)
            output = torch.einsum('bi, ni->bn', energy_output, trunk_output) + self.bias_last
        else:  # MMC
            molecule_output = self.molecule_net(molecule_input)
            if energy_input is None:
                output = torch.einsum('bi, ni->bn', molecule_output, trunk_output) + self.bias_last
            else:
                energy_output = self.energy_net(energy_input)
                branch = torch.cat((molecule_output, energy_output), dim=1)
                output = torch.einsum('bi, ni->bn', branch, trunk_output) + self.bias_last

        return output
    

class DeepCSNetDataset(Dataset):
    """Custom PyTorch Dataset for DeepCSNet training and inference.
    
    This dataset class handles the specialized data structure required by DeepCSNet,
    where different input modalities (molecular features, energy features, coordinate
    features) need to be organized and batched appropriately for the coefficient-subnet
    architecture.
    
    Overview
    --------
    Unlike standard datasets where each sample is independent, DeepCSNet requires:
    
    - Branch inputs (molecule/energy): vary per case, shape [batch_size, features]
    - Trunk inputs (coordinates): shared across all cases, shape [n_points, coord_dim]
    - Targets: cross section values, shape [batch_size, n_points]
    
    This dataset supports two splitting strategies:
    
    1. ``split_by_mole=True``: Iterate over cases (molecules/energies)
    2. ``split_by_mole=False``: Iterate over coordinates
    
    Attributes
    ----------
    molecule_inputs : torch.Tensor or None
        Molecular feature tensor, shape [n_cases, molecule_dim].
    energy_inputs : torch.Tensor or None
        Energy feature tensor, shape [n_cases, energy_dim].
    trunk_inputs : torch.Tensor
        Coordinate feature tensor, shape [n_points, coordinate_dim].
    targets : torch.Tensor or None
        Target cross section tensor, shape [n_cases, n_points].
    split_by_mole : bool
        Splitting strategy flag (True: by cases, False: by coordinates).
    """
    
    def __init__(self,
                 trunk_inputs: torch.Tensor,
                 molecule_inputs: torch.Tensor = None,
                 energy_inputs: torch.Tensor = None,
                 targets: torch.Tensor = None,
                 split_by_mole: bool = True
                ):
        """Initialize the DeepCSNet dataset.
        
        Parameters
        ----------
        trunk_inputs : torch.Tensor
            Coordinate features (output dimensions).
            Shape: [n_points, coordinate_dim]
        molecule_inputs : torch.Tensor, optional
            Molecular features tensor. Shape: [n_cases, molecule_dim].
            Default is None.
        energy_inputs : torch.Tensor, optional
            Energy features tensor. Shape: [n_cases, energy_dim].
            Default is None.
        targets : torch.Tensor, optional
            Target cross section values. Shape: [n_cases, n_points].
            Default is None.
        split_by_mole : bool, default=True
            Splitting strategy:
            - True: Split by case index (iterate over molecules/energies)
            - False: Split by coordinate index (iterate over output points)
        
        Raises
        ------
        ValueError
            If both molecule_inputs and energy_inputs are None.
        """
        if molecule_inputs is None and energy_inputs is None:
            raise ValueError("At least one of molecule_inputs or energy_inputs must be provided for DeepCSNetDataset.")
        self.molecule_inputs = molecule_inputs
        self.energy_inputs = energy_inputs
        self.trunk_inputs = trunk_inputs
        self.targets = targets
        self.split_by_mole = split_by_mole

    def __len__(self):
        """Return the size of the dataset.
        
        Returns
        -------
        int
            Dataset length based on splitting strategy:
            - If split_by_mole=True: number of cases
            - If split_by_mole=False: number of output coordinates
        """
        if self.split_by_mole:
            return len(self.energy_inputs) if self.energy_inputs is not None else len(self.molecule_inputs)
        else:
            return len(self.trunk_inputs)
        
    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Sample index interpretation depends on split_by_mole:
            - If split_by_mole=True: index into cases
            - If split_by_mole=False: index into coordinates
        
        Returns
        -------
        tuple
            A 4-tuple (molecule_input, energy_input, trunk_input, target):
            
            When split_by_mole=True:
            - molecule_input: tensor [molecule_dim] or None
            - energy_input: tensor [energy_dim] or None
            - trunk_input: tensor [n_points, coordinate_dim]
            - target: tensor [n_points]
            
            When split_by_mole=False:
            - molecule_input: tensor [n_cases, molecule_dim] or None
            - energy_input: tensor [n_cases, energy_dim] or None
            - trunk_input: tensor [coordinate_dim]
            - target: tensor [n_cases]
        """
        if self.split_by_mole:
            # Split by the sample indices of molecule_inputs or energy_inputs
            if self.energy_inputs is not None and self.molecule_inputs is not None:
                return self.molecule_inputs[idx, :], self.energy_inputs[idx, :], self.trunk_inputs, self.targets[idx, :]
            elif self.energy_inputs is not None:
                return None, self.energy_inputs[idx, :], self.trunk_inputs, self.targets[idx, :]
            else:
                return self.molecule_inputs[idx, :], None, self.trunk_inputs, self.targets[idx, :]
        else:
            # Split by the sample indices of trunk_inputs
            return self.molecule_inputs, self.energy_inputs, self.trunk_inputs[idx, :], self.targets[:, idx]
        

class DeepCSNetModel(BaseModel):
    """High-level training and inference wrapper for DeepCSNet.

    This class provides a complete pipeline for working with DeepCSNet, including:
    data preparation and batching, training loop, model checkpointing, and inference.
    
    Attributes
    ----------
    network : torch.nn.Module
        The underlying DeepCSNet architecture.
    dataset : DeepCSNetDataset
        Custom dataset instance (set by prepare_train_data).
    dataloader : torch.utils.data.DataLoader
        PyTorch DataLoader with custom collate function.
    loss_func : callable
        Loss function (defaults to MSELoss).
    optimizer : torch.optim.Optimizer
        Optimizer for gradient-based training.
    writer : torch.utils.tensorboard.SummaryWriter or None
        TensorBoard writer for logging.
    checkpoint_dir : str or None
        Directory for saving checkpoints.
    start_epoch : int
        Starting epoch for resuming training.
    """

    def __init__(self, network: nn.Module):
        """Initialize the DeepCSNetModel.
        
        Parameters
        ----------
        network : torch.nn.Module
            The DeepCSNet architecture to be trained.
            Must be an instance of DeepCSNet or compatible network.
        """
        super().__init__(network=network)
        # defaults for optional features
        self.writer = None
        self.checkpoint_dir = None
        self.start_epoch = 0

    
    def prepare_train_data(self,
                           trunk_inputs: torch.Tensor,
                           molecule_inputs: torch.Tensor = None,
                           energy_inputs: torch.Tensor = None,
                           targets: torch.Tensor = None,
                           split_by_mole: bool = True,
                           batch_size: int = None,
                           shuffle: bool = False,
                           drop_last: bool = False,
                        ):
        """Prepare training data and create a DataLoader.
        
        Converts raw tensor data into a DeepCSNetDataset and wraps it with a
        DataLoader that uses a custom collate function.
        
        Parameters
        ----------
        trunk_inputs : torch.Tensor
            Coordinate features (output dimensions).
            Shape: [n_points, coordinate_dim]
        molecule_inputs : torch.Tensor, optional
            Molecular features. Shape: [n_cases, molecule_dim].
            Default is None.
        energy_inputs : torch.Tensor, optional
            Energy features. Shape: [n_cases, energy_dim].
            Default is None.
        targets : torch.Tensor, optional
            Target cross section values. Shape: [n_cases, n_points].
            Default is None.
        split_by_mole : bool, default=True
            Splitting strategy:
            - True: Iterate over cases, trunk_inputs replicated
            - False: Iterate over coordinates, branch inputs replicated
        batch_size : int, optional
            Number of samples per batch. If None, uses full dataset.
            Default is None.
        shuffle : bool, default=False
            Whether to shuffle the dataset at the beginning of each epoch.
        drop_last : bool, default=False
            Whether to drop the last incomplete batch.
        """
        # Move data to the specified device (e.g., GPU)
        if molecule_inputs is not None:
            molecule_inputs = molecule_inputs.to(DEVICE())
        if energy_inputs is not None:
            energy_inputs = energy_inputs.to(DEVICE())
        trunk_inputs = trunk_inputs.to(DEVICE())
        targets = targets.to(DEVICE())
        # Create the dataset
        self.dataset = DeepCSNetDataset(trunk_inputs=trunk_inputs,
                                        molecule_inputs=molecule_inputs,
                                        energy_inputs=energy_inputs,
                                        targets=targets,
                                        split_by_mole=split_by_mole)
        
        # Set batch size to the full dataset size if not specified
        batch_size = len(self.dataset) if batch_size is None else batch_size

        def custom_collate_fn(batch):
            """Custom collate function for DeepCSNet batching.
            
            Standard PyTorch collate stacks all tensors along a new batch dimension.
            However, DeepCSNet requires trunk_inputs to remain [n_points, coord_dim]
            (not batched) since all cases in a batch share the same output coordinates.
            
            Parameters
            ----------
            batch : list of tuples
                List of samples from DeepCSNetDataset.__getitem__().
                Each tuple: (molecule_input, energy_input, trunk_input, target)
            
            Returns
            -------
            tuple
                (molecule_inputs, energy_inputs, trunk_inputs, targets) where:
                - molecule_inputs: [batch_size, molecule_dim] or None
                - energy_inputs: [batch_size, energy_dim] or None
                - trunk_inputs: [n_points, coordinate_dim] (unbatched)
                - targets: [batch_size, n_points]
            """
            molecule_inputs = [item[0] for item in batch]
            energy_inputs = [item[1] for item in batch]
            trunk_inputs = batch[0][2]  # trunk_inputs is the same for all items in the batch
            targets = [item[3] for item in batch]

            # Stack inputs
            molecule_inputs = torch.stack(molecule_inputs) if molecule_inputs[0] is not None else None
            energy_inputs = torch.stack(energy_inputs) if energy_inputs[0] is not None else None
            targets = torch.stack(targets)

            return molecule_inputs, energy_inputs, trunk_inputs, targets
        
        # Create the DataLoader with custom collate function
        self.dataloader = DataLoader(dataset=self.dataset,
                                     shuffle=shuffle,
                                     batch_size=batch_size,
                                     drop_last=drop_last,
                                     collate_fn=custom_collate_fn)
        
    
    def calc_loss(self, data):
        """Calculate the training loss for a given batch.
        
        Performs forward pass and computes loss using configured loss function.
        
        Parameters
        ----------
        data : tuple
            A 4-tuple (molecule_inputs, energy_inputs, trunk_inputs, targets):
            - molecule_inputs: [batch_size, molecule_dim] or None
            - energy_inputs: [batch_size, energy_dim] or None
            - trunk_inputs: [n_points, coordinate_dim]
            - targets: [batch_size, n_points]
        
        Returns
        -------
        torch.Tensor
            Scalar loss value from loss_func(predictions, targets).
        """
        molecule_inputs, energy_inputs, trunk_inputs, targets = data
        predictions = self.network(trunk_input=trunk_inputs, molecule_input=molecule_inputs, energy_input=energy_inputs)
        loss = self.loss_func(predictions, targets)

        return loss
    

    def predict(self, trunk_input, molecule_input=None, energy_input=None):
        """Perform inference using the trained DeepCSNet model.
        
        Sets network to evaluation mode and performs forward pass.
        
        Parameters
        ----------
        trunk_input : torch.Tensor
            Coordinate features for output dimensions.
            Shape: [n_points, coordinate_dim]
        molecule_input : torch.Tensor, optional
            Molecular features for cases. Shape: [batch_size, molecule_dim].
            Required in MMC mode, should be None in SMC mode.
        energy_input : torch.Tensor, optional
            Energy features for cases. Shape: [batch_size, energy_dim].
            Required in SMC mode, optional in MMC mode.
        
        Returns
        -------
        torch.Tensor
            Predicted cross section values.
            Shape: [batch_size, n_points]
        """
        self.network.eval()  # Set the network to evaluation mode
        
        return self.network(trunk_input=trunk_input, molecule_input=molecule_input, energy_input=energy_input)
    

    def train(self,
              num_epochs: int,
              optimizer: torch.optim.Optimizer = None,
              optimizer_cfg: dict = None,
              lr: float = 1e-4,
              lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
              print_loss: bool = True,
              print_loss_freq: int = 1,
              tensorboard_logdir: str = None,
              save_final_model: bool = True,
              final_model_path: str = None,
              checkpoint_dir: str = None,
              checkpoint_freq: int = 1,
              resume_from: str = None,
              ):
        """Train the DeepCSNet model with comprehensive configuration.
        
        Implements a complete training pipeline with flexible optimizer configuration,
        learning rate scheduling, checkpoint management, and TensorBoard logging.
        
        Parameters
        ----------
        num_epochs : int
            Total number of training epochs.
            If resuming: this is TOTAL target epochs (not additional).
            Example: If resume from epoch 500 with num_epochs=1000,
            training continues for 500 more epochs (500→1000).
        optimizer : torch.optim.Optimizer, optional
            Pre-configured optimizer instance.
            If provided, used directly. If None, created based on optimizer_cfg or lr.
            Default is None.
        optimizer_cfg : dict, optional
            Dictionary configuration for building optimizer when optimizer=None.
            Format: ``{'name': 'OptimizerName', 'params': {param_dict}}``
            Example: ``{'name': 'Adam', 'params': {'lr': 1e-4, 'weight_decay': 1e-5}}``
            Supported names: Any optimizer in torch.optim (Adam, SGD, AdamW, etc.)
            Default is None.
        lr : float, default=1e-4
            Default learning rate when creating optimizer automatically.
            Used only if optimizer=None and 'lr' not in optimizer_cfg['params'].
            Typical range: 1e-5 to 1e-3.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler or callable, optional
            Learning rate scheduler for adaptive adjustment.
            Can be PyTorch scheduler or custom callable.
            If provided, .step() is called after each epoch.
            Default is None.
        print_loss : bool, default=True
            Whether to print loss to console during training.
        print_loss_freq : int, default=1
            Frequency (in epochs) for printing loss.
            Example: print_loss_freq=10 prints every 10 epochs.
        tensorboard_logdir : str, optional
            Directory path for TensorBoard log files.
            If provided, creates SummaryWriter and logs loss each epoch.
            View logs with: ``tensorboard --logdir=<tensorboard_logdir>``
            Default is None.
        save_final_model : bool, default=False
            Whether to save final trained model after all epochs.
        final_model_path : str, optional
            File path for saving final model if save_final_model=True.
            If None and checkpoint_dir set: saves to <checkpoint_dir>/final_model.pth
            If None and checkpoint_dir None: saves to './final_model.pth'
            Default is None.
        checkpoint_dir : str, optional
            Directory path for saving periodic training checkpoints.
            If provided, checkpoints named 'checkpoint_epoch_N.pth' are saved.
            Enables training interruption recovery.
            Default is None.
        checkpoint_freq : int, default=1
            Frequency (in epochs) for saving checkpoints.
            Example: checkpoint_freq=100 saves every 100 epochs.
        resume_from : str, optional
            File path to checkpoint for resuming training.
            If provided, loads model state, optimizer state, and epoch number.
            Default is None.
        
        Raises
        ------
        RuntimeError
            If prepare_train_data() has not been called.
        ValueError
            If optimizer_cfg specifies unknown optimizer name.
        FileNotFoundError
            If resume_from path does not exist.
        """
        # Ensure dataloader exists
        if not hasattr(self, 'dataloader') or self.dataloader is None:
            raise RuntimeError("Call prepare_train_data(...) before train(...)")
        
        # Set default loss is not set
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
            for molecule_batch, energy_batch, trunk_batch, target_batch in self.dataloader:
                # Compute loss
                loss = self.calc_loss((molecule_batch, energy_batch, trunk_batch, target_batch))
                
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
            if print_loss and print_loss_freq > 0 and (epoch + 1) % print_loss_freq == 0:
                print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:g}')

            if self.writer is not None:
                self.writer.add_scalar('Loss', avg_loss, epoch + 1)
                self.writer.flush()

            # checkpointing
            if self.checkpoint_dir is not None and checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
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
   
