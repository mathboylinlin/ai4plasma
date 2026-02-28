"""Neural Architecture Search for Physics-Informed Neural Networks (NAS-PINN).

This module implements the NAS-PINN framework for automatically searching the optimal
architecture of Physics-Informed Neural Networks (PINNs) to solve partial differential
equations (PDEs). It uses a differentiable architecture search method within a relaxed
search space to find network architectures that balance accuracy and computational efficiency.

NAS-PINN Classes
----------------
- `NasPINN`: Main class implementing the architecture search framework.

NAS-PINN References
-------------------
[1] Y. Wang, L. Zhong, "NAS-PINN: Neural architecture search-guided physics-informed neural
    network for solving PDEs," Journal of Computational Physics, vol. 496, p. 112603, 2024.
"""


import os
from tqdm import tqdm
from typing import Dict
import torch
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.piml.pinn import PINN, VisualizationCallback
from ai4plasma.config import DEVICE


class NasPINN:
    """
    Neural Architecture Search for Physics-Informed Neural Networks (NAS-PINN).
    
    This class implements the NAS-PINN framework for automated architecture search in 
    Physics-Informed Neural Networks (PINNs). It performs differentiable architecture search 
    to find the optimal network architecture for solving partial differential equations (PDEs) 
    within a given search space. The framework combines bi-level optimization: inner loop for 
    weight adaptation and outer loop for architecture parameter optimization.
    
    Attributes
    ----------
    pinn_model : PINN
        The PINN model instance with relaxed FNN structure and searchable architecture parameters.
    writer : SummaryWriter, optional
        TensorBoard writer for logging training metrics. Initialized during search process.
    history : dict
        Dictionary storing training history, including 'search_loss' trajectory.
    visualization_callback : dict
        Dictionary storing visualization callbacks indexed by name.
    last_outer_epochs : int
        Number of completed outer loop iterations (useful for resuming training).
    outer_epochs : int
        Target total number of outer loop iterations.
    inner_epochs : int
        Number of inner loop iterations per outer loop step.
    outer_opt : torch.optim.Optimizer
        Optimizer for outer loop (architecture parameter updates).
    inner_opt : torch.optim.Optimizer
        Optimizer for inner loop (network weight updates).
    """
    def __init__(self, pinn_model: PINN):
        """
        Initialize the NAS-PINN framework.
        
        Initializes the NAS-PINN instance with a given PINN model configured for architecture
        search. Sets up optimizers for bi-level optimization and initializes tracking dictionaries
        for training history and visualization callbacks.
        
        Parameters
        ----------
        pinn_model : PINN
            A Physics-Informed Neural Network model instance with:
            - A relaxed FNN structure supporting architecture search
            - Architecture parameters (g) that can be optimized during training
            - Methods: calc_loss(), calc_loss_archi(), and network attributes
        """
        self.pinn_model = pinn_model

        self.writer = None  # TensorBoard writer (initialized in search)
        self.history = {
            'search_loss': [],  # NAS search loss history
        }

        self.visualization_callback: Dict[str, VisualizationCallback] = {}

        # Training state tracking for resumable training
        self.last_outer_epochs = 0  # Last completed epoch (for resuming)
        self.outer_epochs = 0       # Total target epochs
        self.inner_epochs = 0       # Inner loop iterations per task
        self.outer_opt = torch.optim.Adam(self.pinn_model.network.arch_parameters(), lr=1e-5)
        self.inner_opt = torch.optim.Adam(self.pinn_model.network.parameters(), lr=1e-4)


    def load_nas_model(self, checkpoint_path: str):
        """
        Load NAS-PINN model and training state from checkpoint.
        
        Restores the NAS parameters, architecture parameters, and optimizer states from a
        previously saved checkpoint file. This enables resuming interrupted training from
        the exact point where it was saved.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (.pth format).
        """
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE())

        # Restore relaxed FNN parameters and architecture parameters
        self.pinn_model.network.load_state_dict(checkpoint['nas_state_dict']['network'])
        self.pinn_model.network.load_gs(checkpoint['nas_state_dict']['arch_param'])

        # Restore training state
        self.last_outer_epochs = checkpoint.get('outer_epochs', 0)
        self.inner_epochs = checkpoint.get('inner_epochs', 0)
        self.outer_opt.load_state_dict(checkpoint['outer_opt'])
        self.inner_opt.load_state_dict(checkpoint['inner_opt'])

        print(f"NAS-PINN model loaded from {checkpoint_path} at epoch {self.last_outer_epochs}.")

    
    def save_nas_model(self, epoch: int, checkpoint_path: str):
        """
        Save NAS-PINN model and training state to checkpoint.
        
        Saves the current network parameters, architecture parameters, and optimizer states
        to enable training resumption. Checkpoints are essential for long-running architecture
        searches that may be interrupted.
        
        Parameters
        ----------
        epoch : int
            Current epoch number in the outer loop (for tracking and logging progress).
        checkpoint_path : str
            Path where the checkpoint file should be saved (.pth format).
        """
        checkpoint = {
            'nas_state_dict': {
                'network': self.pinn_model.network.state_dict(),
                'arch_param': self.pinn_model.network.arch_parameters(),
            },
            'outer_epochs': epoch,
            'inner_epochs': self.inner_epochs,
            'outer_opt': self.outer_opt.state_dict(),
            'inner_opt': self.inner_opt.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"NAS-PINN model saved at {checkpoint_path}.")


    def search(self,
               outer_epochs: int,
               inner_epochs: int,
               outer_opt: torch.optim.Optimizer = None,
               inner_opt: torch.optim.Optimizer = None,
               print_freq: int = 10,
               tensorboard_logdir: str = None,
               log_freq: int = 50,
               checkpoint_dir: str = None,
               checkpoint_freq: int = 100,
               load_from_checkpoint: str = None,
               final_model_path: str = None):
        """
        Search for optimal architecture parameters using differentiable search.
        
        Executes the NAS-PINN algorithm with bi-level optimization:
        
        - **Inner Loop**: Updates network weights using calc_loss() with fixed
          architecture parameters (g).
        - **Outer Loop**: Updates architecture parameters using calc_loss_archi()
          with the adapted network weights.
        
        Parameters
        ----------
        outer_epochs : int
            Number of search iterations (outer loop). Typical range depends on PDE
            complexity: 500-500,000 for different problems.
        inner_epochs : int
            Number of gradient steps for weight adaptation per outer epoch (inner
            loop). Typical range: 1-20 steps, controls inner loop optimization depth.
        outer_opt : torch.optim.Optimizer, optional
            Optimizer for outer loop architecture parameter updates. If None,
            defaults to Adam(lr=1e-5).
        inner_opt : torch.optim.Optimizer, optional
            Optimizer for inner loop weight updates. If None,
            defaults to Adam(lr=1e-4).
        print_freq : int, default=10
            Print training loss statistics every print_freq epochs to console.
        tensorboard_logdir : str, optional
            Directory for TensorBoard event logs. If None, TensorBoard logging is
            disabled. Create logs at specified interval (log_freq) for performance
            monitoring.
        log_freq : int, default=50
            Log 'Loss' and 'Loss-archi' metrics to TensorBoard every log_freq epochs.
        checkpoint_dir : str, optional
            Directory to save periodic checkpoints. If None, no checkpoints are
            saved. Directory is created if it doesn't exist.
        checkpoint_freq : int, default=100
            Save checkpoint every checkpoint_freq epochs for training resumption.
        load_from_checkpoint : str, optional
            Path to checkpoint file for resuming interrupted training. If provided,
            restores network parameters, architecture parameters, optimizer states,
            and training epoch count from checkpoint.
        final_model_path : str, optional
            Path to save the final model after completing architecture search.
            If provided, the best model (final network state with architecture
            parameters) will be saved at this location.
        
        Returns
        -------
        None
            Training history is stored in self.history['search_loss'].
            Network and architecture parameters are updated in-place.
            Final network architecture can be extracted via
            self.pinn_model.network.searched_neuron().
        """

        if outer_opt is not None:
            self.outer_opt = outer_opt
        if inner_opt is not None:
            self.inner_opt = inner_opt

        if load_from_checkpoint:
            self.load_nas_model(load_from_checkpoint)
        else:
            self.last_outer_epochs = 0

        self.outer_epochs = self.last_outer_epochs + outer_epochs
        self.inner_epochs = inner_epochs

        # Setup TensorBoard
        if tensorboard_logdir:
            self.writer = SummaryWriter(tensorboard_logdir)
            self.pinn_model.writer = self.writer  # Pass writer to PINN model for visualization callbacks

        # Create checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # NAS-PINN loop
        for epoch_arch in range(self.last_outer_epochs, self.outer_epochs):

            loop = tqdm(range(self.inner_epochs), total=self.inner_epochs)
            for index in loop:
                
                self.inner_opt.zero_grad()
                loss, _ = self.pinn_model.calc_loss()
                loss.backward(retain_graph=True)
                self.inner_opt.step()
                loop.set_description("Epoch %d" % (epoch_arch+1))
            
            # Outer loop: arch search, inner loop: weight update
            self.outer_opt.zero_grad()
            loss_archi, _ = self.pinn_model.calc_loss_archi()
            loss_archi.backward()
            self.outer_opt.step()

            # Training log
            loss_val = loss.item()
            loss_archi_val = loss_archi.item()
            if(epoch_arch+1)%log_freq == 0:
                self.writer.add_scalar('Loss-archi', loss_archi_val, epoch_arch)
                self.writer.add_scalar('Loss', loss_val, epoch_arch)
                self.writer.flush()
            
            if (epoch_arch+1)%print_freq == 0:
                print('Epoch: [%d/%d] Loss: %g Loss-archi: %g' % (epoch_arch+1, outer_epochs, loss_val, loss_archi_val))
            
                final_neuron = self.pinn_model.network.searched_neuron()
                print(f"Current Architecture (Epoch {epoch_arch+1}): {final_neuron}")
            
            # Execute visualization callbacks
            loss_dict = {'Loss': loss, 'Loss-archi': loss_archi}
            self.pinn_model._execute_visualization_callbacks(epoch_arch, loss_dict=loss_dict, total_loss=loss)
            
            # Save checkpoint periodically
            if checkpoint_dir and (epoch_arch+1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'nas_pinn_epoch_{epoch_arch+1}.pth')
                self.save_nas_model(epoch_arch+1, checkpoint_path)
        
        # Save final model if path is specified
        if final_model_path:
            # If final_model_path is a directory, use default filename
            if os.path.isdir(final_model_path) or final_model_path.endswith(os.sep):
                final_model_path = os.path.join(final_model_path, 'nas_pinn_final.pth')
            
            # Create directories if they don't exist
            model_dir = os.path.dirname(final_model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            
            self.save_nas_model(self.outer_epochs, final_model_path)
            print(f"Final NAS-PINN model saved to {final_model_path}")
        
        # Print final searched architecture
        print("\n" + "="*80)
        print(f"NAS-PINN Architecture Search Completed!")
        print(f"Total Outer Epochs: {self.outer_epochs}")
        final_architecture = self.pinn_model.network.searched_neuron()
        print(f"Final Searched Architecture: {final_architecture}")
        print("="*80 + "\n")