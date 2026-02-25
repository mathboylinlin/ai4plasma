"""Meta-learning Physics-Informed Neural Networks (Meta-PINN).

This module implements Model-Agnostic Meta-Learning (MAML) for physics-informed
neural networks. It provides a task abstraction, bi-level optimization pipeline,
and training utilities for rapid adaptation to new physics tasks with limited data.

Meta-PINN Classes
-----------------
- `MetaTask`: Abstract task interface for meta-learning.
- `PINNTask`: PINN-specific task implementation with equation term losses.
- `MetaPINN`: MAML trainer for PINN task batches.

Meta-PINN References
--------------------
[1] L. Zhong, B. Wu, and Y. Wang, "Accelerating physics-informed neural network
    based 1D arc simulation by meta learning," Journal of Physics D: Applied Physics,
    vol. 56, p. 074006, 2023.
"""

import os
import traceback
from typing import Callable, Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.core.network import FNN
from ai4plasma.piml.geo import Geo1D
from ai4plasma.piml.pinn import PINN, VisualizationCallback
from ai4plasma.piml.cs_pinn import calc_GL_coefs
from ai4plasma.plasma.prop import ArcPropSpline
from ai4plasma.utils.math import df_dX
from ai4plasma.config import REAL, DEVICE


class MetaTask(ABC):
    """
    Abstract base class for defining meta-learning tasks.

    A meta-learning task encapsulates a specific problem instance within a broader
    family of related problems. In the MAML framework, each task contains support
    and query sets used for task-specific adaptation and meta-parameter updates.

    This abstraction allows the meta-learning framework to handle diverse physics
    problems in a unified manner by implementing the compute_loss method. It follows
    the Strategy design pattern, enabling different task types to be plugged into the
    MetaPINN framework without modifying core algorithms.

    Parameters
    ----------
    task_id : str
        Unique identifier for this task (e.g., 'Arc_I=200A', 'Poisson_k=1.5')
    support_data : Dict[str, torch.Tensor], optional
        Dictionary mapping data names to support set tensors. Support data is used
        for task-specific adaptation during the inner loop. Default is empty dict.
        Example: {'Domain': x_domain, 'Boundary': x_boundary}
    query_data : Dict[str, torch.Tensor], optional
        Dictionary mapping data names to query set tensors. Query data is disjoint
        from support data and used for meta-parameter updates. Default is empty dict.
        Same structure as support_data but with different samples.

    Attributes
    ----------
    task_id : str
        Unique identifier for this task instance.
    support_data : Dict[str, torch.Tensor]
        Support set for inner loop training (few-shot adaptation).
    query_data : Dict[str, torch.Tensor]
        Query set for outer loop meta-updates (meta-gradient computation).
    """

    def __init__(self, 
                 task_id: str,
                 support_data: Dict[str, torch.Tensor] = None,
                 query_data: Dict[str, torch.Tensor] = None):
        """
        Initialize a meta-learning task.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task instance.
        support_data : Dict[str, torch.Tensor], optional
            Support set data for inner loop adaptation. Keys are equation term names
            (e.g., 'Domain', 'Boundary'). Values are collocation point tensors.
            Default is None (converted to empty dict).
        query_data : Dict[str, torch.Tensor], optional
            Query set data for meta-validation. Must have same structure as
            support_data but with different samples. Default is None.
        """
        super().__init__()

        self.task_id = task_id
        self.support_data = support_data or {}
        self.query_data = query_data or {}

    @abstractmethod
    def compute_loss(self, network: nn.Module, 
                     data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss for this task on given data.

        This abstract method defines how to evaluate the network's performance on task data.
        It is called during both inner loop adaptation (support set) and outer loop
        meta-updates (query set). Subclasses must implement this method.

        Parameters
        ----------
        network : nn.Module
            Neural network to evaluate. May be adapted in inner loop.
        data_dict : Dict[str, torch.Tensor]
            Dictionary of data tensors for different equation terms.
            Keys match equation term names (e.g., 'Domain', 'Boundary').
            Values are collocation points or boundary data.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar tensor representing the total weighted loss.
        loss_dict : Dict[str, torch.Tensor]
            Dictionary mapping equation term names to individual losses.
            Useful for monitoring training and statistical analysis.
        """
        pass

    def get_task_id(self) -> str:
        """
        Get unique identifier for this task.

        Returns
        -------
        str
            Task identifier (e.g., 'Arc_I=200A_R=0.01m').
        """
        return self.task_id


    def get_support_data(self) -> Dict[str, torch.Tensor]:
        """
        Get support set data for inner loop training.

        The support set is used to adapt the meta-initialized network to a
        specific task during the inner loop of MAML. Typically contains fewer
        samples than traditional PINN training.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping equation term names to support data tensors.
        """
        return self.support_data
    

    def get_query_data(self) -> Dict[str, torch.Tensor]:
        """
        Get query set data for meta-validation.

        The query set is used to evaluate the adapted network and compute
        gradients for meta-parameter updates in the outer loop. It must be
        disjoint from the support set to ensure unbiased meta-learning.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping equation term names to query data tensors.
        """
        return self.query_data


class PINNTask(MetaTask):
    """
    PINN-specific meta-learning task implementation.

    This class bridges the PINN framework with the meta-learning paradigm by wrapping
    a PINN model and its equation terms into a task suitable for MAML. It implements
    the compute_loss method by iterating over all equation terms defined in the PINN.

    Parameters
    ----------
    task_id : str
        Unique identifier for this physics task.
    pinn_model : PINN, optional
        PINN model instance containing equation definitions.
        Must have defined equation terms via add_equation().
    support_data : Dict[str, torch.Tensor], optional
        Support set collocation points for inner loop adaptation.
    query_data : Dict[str, torch.Tensor], optional
        Query set collocation points for meta-update evaluation.

    Attributes
    ----------
    pinn_model : PINN
        The underlying PINN model defining equation terms and residuals.
    loss_func : callable
        Loss function for comparing residuals to zero (e.g., MSE, smooth L1).
    """
    
    def __init__(self, task_id: str, 
                 pinn_model: Optional[PINN] = None,
                 support_data: Dict[str, torch.Tensor] = None,
                 query_data: Dict[str, torch.Tensor] = None):
        """
        Initialize a PINN task for meta-learning.

        Parameters
        ----------
        task_id : str
            Unique identifier for this physics task.
        pinn_model : PINN, optional
            PINN model instance containing equation definitions.
            Must have defined equation terms via add_equation().
        support_data : Dict[str, torch.Tensor], optional
            Support set collocation points for inner loop adaptation.
        query_data : Dict[str, torch.Tensor], optional
            Query set collocation points for meta-update evaluation.
        """
        super().__init__(task_id, support_data, query_data)
        self.pinn_model = pinn_model
        self.loss_func = pinn_model.loss_func if pinn_model else F.mse_loss

    
    def compute_loss(self, network: nn.Module, 
                     data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total weighted loss for this PINN task.

        Parameters
        ----------
        network : nn.Module
            Neural network to evaluate (may be adapted in inner loop).
        data_dict : Dict[str, torch.Tensor]
            Dictionary mapping equation term names to data tensors.
            Example: {'Domain': x_pde, 'Boundary': x_bc, 'Initial': x_ic}

        Returns
        -------
        total_loss : torch.Tensor
            Scalar weighted sum of all equation term losses.
        loss_dict : Dict[str, torch.Tensor]
            Dictionary of individual losses for monitoring.
            Keys: equation term names, Values: unweighted loss tensors.
        """
        equation_terms = list(self.pinn_model.equation_terms.values())
        loss_dict = {}
        weighted_losses = []
        
        for eq_term in equation_terms:
            # Skip terms without corresponding data
            if eq_term.name not in data_dict:
                continue
            
            # Compute residual for this equation term
            batch_data = data_dict[eq_term.name]
            residual = eq_term.compute_residual(network, batch_data)
            
            # Compute loss (residual should be zero for physics satisfaction)
            target = torch.zeros_like(residual)
            loss = self.loss_func(residual, target)
            loss_dict[eq_term.name] = loss
            
            # Weight and accumulate for total loss
            weighted_losses.append(eq_term.weight * loss)
        
        # Sum all weighted losses
        total_loss = sum(weighted_losses) if weighted_losses else torch.tensor(0.0, dtype=REAL('torch'))

        return total_loss, loss_dict
    

class MetaPINN:
    """
    Meta-Learning Framework for Physics-Informed Neural Networks using MAML.

    This class implements the Model-Agnostic Meta-Learning (MAML) algorithm
    specifically designed for physics-informed neural networks. It learns optimal
    network initializations across multiple related physics tasks, enabling rapid
    adaptation to new tasks with minimal fine-tuning.

    Parameters
    ----------
    train_tasks : List[PINNTask]
        List of PINN tasks for meta-training. Each task should have support and
        query data defined.
    test_tasks : List[PINNTask], optional
        List of PINN tasks for meta-testing/evaluation.

    Attributes
    ----------
    train_tasks : List[PINNTask]
        Training tasks for meta-learning.
    test_tasks : List[PINNTask]
        Test tasks for meta-evaluation.
    meta_network : nn.Module
        The meta-network whose parameters are meta-learned.
    loss_func : callable
        Loss function for computing residuals (default: smooth_l1_loss).
    writer : SummaryWriter
        TensorBoard writer for logging meta-training progress.
    history : Dict
        Training history storing meta-train losses.
    visualization_callbacks : Dict[str, VisualizationCallback]
        Registered callbacks for real-time visualization.
    outer_epochs : int
        Number of meta-training iterations completed.
    inner_epochs : int
        Number of adaptation steps per task (inner loop).
    outer_lr : float
        Meta-learning rate (outer loop, typically 1e-4 to 1e-3).
    inner_lr : float
        Task adaptation learning rate (inner loop, typically 1e-5 to 1e-3).
    beta1, beta2 : float
        Adam optimizer momentum parameters (default 0.9, 0.999).
    epsilon : float
        Adam optimizer numerical stability constant (default 1e-8).
    """

    def __init__(self, train_tasks: List[PINNTask], test_tasks: List[PINNTask] = None):
        """
        Initialize Meta-PINN framework with training tasks.

        Parameters
        ----------
        train_tasks : List[PINNTask]
            List of PINN tasks for meta-training. Each task should have support
            and query data defined.
        test_tasks : List[PINNTask], optional
            List of PINN tasks for meta-testing/evaluation.
        """
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        # Initialize meta-network from first task's PINN model
        self.meta_network = train_tasks[0].pinn_model.network if train_tasks else None

        self.loss_func = F.smooth_l1_loss

        self.writer = None  # TensorBoard writer (initialized in meta_train)
        self.history = {
            'meta_train_loss': [],  # Meta-training loss history
        }

        self.visualization_callbacks: Dict[str, VisualizationCallback] = {}
        
        # Training state tracking for resumable training
        self.last_outer_epochs = 0  # Last completed epoch (for resuming)
        self.outer_epochs = 0       # Total target epochs
        self.inner_epochs = 0       # Inner loop iterations per task
        self.outer_lr = 1e-4        # Meta-learning rate (outer loop)
        self.inner_lr = 1e-5        # Task adaptation learning rate (inner loop)
        self.beta1 = 0.9            # Adam momentum parameter 1
        self.beta2 = 0.999          # Adam momentum parameter 2
        self.epsilon = 1e-8         # Adam numerical stability constant


    def load_meta_model(self, checkpoint_path: str):
        """
        Load meta-model from checkpoint for resuming training.

        This method restores the meta-network parameters and training state from
        a saved checkpoint, enabling interrupted meta-training to resume seamlessly
        from the last saved epoch.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (.pth). Should contain:
            - meta_network_state_dict: Learned meta-parameters
            - outer_epochs: Last completed meta-training epoch
            - inner_epochs: Inner loop iteration count
            - outer_lr, inner_lr: Learning rates
            - beta1, beta2, epsilon: Adam optimizer hyperparameters
        """
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE())

        # Restore meta-network parameters
        self.meta_network.load_state_dict(checkpoint['meta_network_state_dict'])
        
        # Restore training state
        self.last_outer_epochs = checkpoint.get('outer_epochs', 0)
        self.inner_epochs = checkpoint.get('inner_epochs', 0)
        self.outer_lr = checkpoint.get('outer_lr', self.outer_lr)
        self.inner_lr = checkpoint.get('inner_lr', self.inner_lr)
        self.beta1 = checkpoint.get('beta1', self.beta1)
        self.beta2 = checkpoint.get('beta2', self.beta2)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        
        print(f"Meta-PINN model loaded from {checkpoint_path} at epoch {self.last_outer_epochs}.")

    
    def save_meta_model(self, epoch: int, checkpoint_path: str):
        """
        Save meta-model checkpoint for later resumption.

        This method saves the current meta-network parameters and training state
        to disk, allowing meta-training to be resumed from this point if interrupted.
        Checkpoints are typically saved at regular intervals during training.

        Parameters
        ----------
        epoch : int
            Current meta-training epoch number.
        checkpoint_path : str
            Path where to save the checkpoint file.
        """
        
        torch.save({
            'meta_network_state_dict': self.meta_network.state_dict(),
            'outer_epochs': epoch,
            'inner_epochs': self.inner_epochs,
            'outer_lr': self.outer_lr,
            'inner_lr': self.inner_lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
        }, checkpoint_path)
        
        print(f"Meta-PINN model saved to {checkpoint_path}.")

    
    def meta_train(self, 
                   outer_epochs: int,
                   inner_epochs: int = 5,
                   outer_lr: float = 1e-4,
                   inner_lr: float = 1e-5,
                   beta1: float = 0.9,
                   beta2: float = 0.999,
                   epsilon: float = 1e-8,
                   print_freq: int = 10,
                   tensorboard_logdir: str = None,
                   log_freq: int = 50,
                   checkpoint_dir: str = None,
                   checkpoint_freq: int = 100,
                   load_from_checkpoint: str = None):
        """
        Execute meta-training using Model-Agnostic Meta-Learning (MAML).

        This method implements the bi-level optimization algorithm of MAML:
        - Inner Loop: Fast adaptation to individual tasks using gradient descent
        - Outer Loop: Meta-parameter update using accumulated query losses

        The goal is to learn an initialization that enables rapid adaptation to new
        physics tasks with minimal fine-tuning. After meta-training, the learned
        initialization can be applied to unseen tasks for few-shot learning.

        Parameters
        ----------
        outer_epochs : int
            Number of meta-training iterations (outer loop).
            Typical range: 500-5000 depending on task complexity.
        inner_epochs : int, default=5
            Number of gradient steps for task adaptation (inner loop).
            Typical range: 1-20 steps.
        outer_lr : float, default=1e-4
            Meta-learning rate for outer loop.
            Controls update speed of meta-parameters.
            Typical range: 1e-5 to 1e-3.
        inner_lr : float, default=1e-5
            Task adaptation learning rate for inner loop.
            Typical range: 1e-6 to 1e-3.
        beta1 : float, default=0.9
            Adam momentum parameter for first moment estimation.
        beta2 : float, default=0.999
            Adam momentum parameter for second moment estimation.
        epsilon : float, default=1e-8
            Adam numerical stability constant.
        print_freq : int, default=10
            Print training progress every print_freq epochs.
        tensorboard_logdir : str, optional
            Directory for TensorBoard logging. If None, logging is disabled.
        log_freq : int, default=50
            Log metrics to TensorBoard every log_freq epochs.
        checkpoint_dir : str, optional
            Directory to save checkpoints during training. If None, no saves.
        checkpoint_freq : int, default=100
            Save checkpoint every checkpoint_freq epochs.
        load_from_checkpoint : str, optional
            Path to checkpoint file for resuming training.
        """
        
        if load_from_checkpoint:
            self.load_meta_model(load_from_checkpoint)
        else:
            self.last_outer_epochs = 0

        self.outer_epochs = self.last_outer_epochs + outer_epochs
        self.inner_epochs = inner_epochs
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Setup TensorBoard
        if tensorboard_logdir:
            self.writer = SummaryWriter(tensorboard_logdir)
        
        # Create checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize Adam optimizer moments for meta-parameters
        # Momentum (first moment): running average of gradients
        momentum = [0 for _, _ in enumerate(self.meta_network.parameters())]
        # Moving average (second moment): running average of squared gradients
        moving_avg = [0 for _, _ in enumerate(self.meta_network.parameters())]

        # Initialize meta-weights (will be updated after each meta-iteration)
        meta_weights = [param.data for param in self.meta_network.parameters()]

        # Meta-training loop (outer loop)
        for epoch in range(self.last_outer_epochs, self.outer_epochs):
            loss_qry_tot = 0  # Initialize total query loss for this meta-iteration
            task_weights = [param.data for param in self.meta_network.parameters()]  # Store meta-parameters before inner loop
            
            loss_qry_task = []  # Track individual task query losses for monitoring

            # Inner loop: Adapt to each task using support set
            for task in self.train_tasks:

                # Reset meta-network to initial meta-parameters for this task
                for kk, param in enumerate(self.meta_network.parameters()):
                    param.data = task_weights[kk]  # Start inner loop from meta-initialization

                # Inner loop step 0: Compute initial support loss and gradient
                loss_spt, _ = task.compute_loss(self.meta_network, task.get_support_data())

                # Compute gradient of support loss w.r.t. parameters
                grad = torch.autograd.grad(loss_spt, self.meta_network.parameters())
                
                # One-step gradient descent for task adaptation
                new_weights = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, task_weights)))

                # Update network with adapted parameters
                for kk, param in enumerate(self.meta_network.parameters()):
                    param.data = new_weights[kk]

                # Inner loop steps 1 to (inner_epochs-1): Continue adaptation
                for j in range(1, inner_epochs):
                    # Compute support loss with updated parameters
                    loss_spt, _ = task.compute_loss(self.meta_network, task.get_support_data())
                    
                    # Compute gradient and update weights
                    grad = torch.autograd.grad(loss_spt, self.meta_network.parameters())
                    new_weights = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, new_weights)))
                    
                    # Apply updated weights to network
                    for kk, param in enumerate(self.meta_network.parameters()):
                        param.data = new_weights[kk]
                
                # Evaluate adapted network on query set (for meta-update)
                loss_qry, _ = task.compute_loss(self.meta_network, task.get_query_data())

                # Track individual task query loss (for monitoring)
                loss_qry_task.append(loss_qry.item()/len(self.train_tasks))

                # Accumulate total query loss across all tasks
                loss_qry_tot += loss_qry
                
                # Outer loop: Compute meta-gradient from query loss
                # This is the key step in MAML: gradient of query loss w.r.t. meta-parameters
                update_grad = torch.autograd.grad(loss_qry, self.meta_network.parameters(), retain_graph=True)
                
                # Adam optimizer update for meta-parameters
                for kk, param in enumerate(meta_weights):
                    # Update first moment (momentum)
                    momentum[kk] = beta1*momentum[kk] + (1 - beta1)*update_grad[kk]
                    # Update second moment (adaptive learning rate)
                    moving_avg[kk] = beta2*moving_avg[kk] + (1 - beta2)*update_grad[kk]**2
                    # Bias-corrected moments
                    corr_momentum = momentum[kk]/(1 - beta1**(epoch+1))
                    corr_moving_avg = moving_avg[kk]/(1 - beta2**(epoch+1))
                    # Adam update rule
                    meta_weights[kk] = meta_weights[kk] - outer_lr*corr_momentum/(torch.sqrt(corr_moving_avg) + epsilon)
                           

            # Average query loss across all tasks (meta-objective)
            loss_qry_tot /= len(self.train_tasks)
            self.history['meta_train_loss'].append(loss_qry_tot.item())
            
            # Print progress at specified intervals
            if (epoch+1) % print_freq == 0:
                print('[%d/%d] Meta-Loss: %g' % (epoch+1, self.outer_epochs, loss_qry_tot.item()))

            # Apply updated meta-parameters to meta-network
            for kk, param in enumerate(self.meta_network.parameters()):
                param.data = meta_weights[kk]

            # Log to TensorBoard
            if self.writer and log_freq > 0 and (epoch + 1) % log_freq == 0:
                self.writer.add_scalar('Meta-Loss', loss_qry_tot.item(), epoch+1)
                self.writer.flush()
            
            # Save checkpoint periodically
            if checkpoint_dir and (epoch+1) % checkpoint_freq == 0:
                self.save_meta_model(epoch+1, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            # Execute visualization callbacks
            self._execute_visualization_callbacks(
                epoch,
                train_task=task,
                meta_train_loss=loss_qry_tot.item(),
            )


        if self.writer:
            self.writer.close()


    def meta_test(self,
                  test_tasks: List[PINNTask],
                  results_dir: str = None):
        """
        Evaluate meta-learned initialization on new test tasks.

        This method demonstrates the meta-learning benefit by initializing the network
        with learned meta-parameters and performing few-shot adaptation on new tasks
        with minimal training data.

        Parameters
        ----------
        test_tasks : List[PINNTask]
            List of new tasks for meta-testing. Each task should have support
            data defined for few-shot adaptation.
        results_dir : str, optional
            Directory to save adapted model checkpoints. One checkpoint per task
            will be saved if provided.
        """
        self.test_tasks = test_tasks

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            save_test_task_model_file = [os.path.join(results_dir, f'{task.get_task_id()}_meta_model.pth') for task in test_tasks]

        # Store meta-parameters for reinitialization across tasks
        task_weights = [param.data for param in self.meta_network.parameters()]

        # Adapt to each test task independently
        for i, task in enumerate(self.test_tasks):
            
            # Reset to meta-initialization for this task
            for kk, param in enumerate(self.meta_network.parameters()):
                param.data = task_weights[kk]

            # Few-shot adaptation: First gradient step on support set
            loss_spt, _ = task.compute_loss(self.meta_network, task.get_support_data())
            grad = torch.autograd.grad(loss_spt, self.meta_network.parameters())
            new_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, self.meta_network.parameters())))

            for kk, param in enumerate(self.meta_network.parameters()):
                param.data = new_weights[kk]

            # Continue adaptation for remaining inner loop iterations
            for j in range(1, self.inner_epochs):
                loss_spt, _ = task.compute_loss(self.meta_network, task.get_support_data())
                grad = torch.autograd.grad(loss_spt, self.meta_network.parameters())
                new_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, new_weights)))
                for kk, param in enumerate(self.meta_network.parameters()):
                    param.data = new_weights[kk]
            
            # Save adapted parameters for this test task
            torch.save({'network_state_dict': self.meta_network.state_dict()}, save_test_task_model_file[i])

    
    def register_visualization_callback(self, callback: VisualizationCallback):
        """
        Register a visualization callback for training monitoring.

        Parameters
        ----------
        callback : VisualizationCallback
            Callback instance to register (from pinn.VisualizationCallback).
        """
        self.visualization_callbacks[callback.name] = callback
        print(f"Registered visualization callback: {callback.name} (log_freq={callback.log_freq})")
    
    def _execute_visualization_callbacks(self, epoch: int, **kwargs):
        """
        Execute all registered visualization callbacks.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        kwargs : dict
            Additional information to pass to callbacks.
        """
        if not self.writer or len(self.visualization_callbacks) == 0:
            return
        
        # Add meta_pinn reference to kwargs for callbacks that need it
        kwargs['meta_pinn'] = self
        
        for callback_name, callback in self.visualization_callbacks.items():
            # Check if it's time to execute this callback
            if callback.log_freq > 0 and (epoch + 1) % callback.log_freq == 0:
                try:
                    # Execute callback visualization (pass network instead of meta_pinn)
                    # This maintains compatibility with PINN.VisualizationCallback interface
                    figures = callback.visualize(self.meta_network, epoch, self.writer, **kwargs)
                    
                    if figures and isinstance(figures, dict):
                        # Log figures to TensorBoard
                        for plot_name, fig in figures.items():
                            if hasattr(fig, 'savefig'):  # Check if it's a matplotlib figure
                                self.writer.add_figure(
                                    f'Visualization/{callback_name}/{plot_name}',
                                    fig,
                                    global_step=epoch
                                )
                                plt.close(fig)  # Close figure to save memory
                except Exception as e:
                    print(f"Error in visualization callback '{callback_name}': {str(e)}")
                    
                    traceback.print_exc()


class MetaStaArc1DNet(nn.Module):
    """
    Neural network wrapper for meta-learning 1D stationary arc plasma problems.

    Parameters
    ----------
    network : nn.Module
        Backbone neural network (e.g., FNN) that maps r → N(r).
        Input shape: [batch_size, 1] (normalized radius)
        Output shape: [batch_size, 1] (log temperature)
    """
    def __init__(self, network):
        super(MetaStaArc1DNet, self).__init__()

        self.network = network  # Backbone network (maps r → log T)

    def forward(self, x):
        """
        Forward pass with exponential activation.

        Parameters
        ----------
        x : torch.Tensor
            Input normalized radius, shape [batch_size, 1]

        Returns
        -------
        torch.Tensor
            Normalized temperature (always positive), shape [batch_size, 1].
        """
        out = self.network(x)  # Backbone output (log-space)
        out = torch.exp(out)   # Convert to positive temperature
        return out


class MetaStaArc1DModel(PINN):
    """
    PINN model for 1D stationary arc adapted for meta-learning.

    This class implements a specialized PINN for solving the 1D steady-state arc
    plasma equation in a meta-learning context. Unlike the standard StaArc1DModel, 
    this version is designed as a task within the MetaPINN framework for learning 
    across multiple arc configurations.

    Parameters
    ----------
    R : float
        Arc radius in meters (e.g., 0.01 for 10mm).
    I : float
        Arc current in amperes (e.g., 100, 150, 200).
    Tb : float, default=300.0
        Boundary temperature at r=R in Kelvin.
    T_red : float, default=1e4
        Temperature normalization constant (typically 10000K).
    backbone_net : nn.Module, default=FNN([1,100,100,100,100,1])
        Backbone neural network for temperature prediction.
    train_data_size : int, default=500
        Number of collocation points for training.
    test_data_size : int, default=501
        Number of points for evaluation/prediction.
    sample_mode : str, default='uniform'
        Sampling strategy ('uniform', 'random', or 'lhs').
    GL_degree : int, default=100
        Degree of Gauss-Legendre quadrature for arc conductance integration.
    prop : ArcPropSpline
        Material property splines (κ, σ, ε_nec as functions of T).
    """

    def __init__(
        self,
        R,
        I,
        Tb=300.0,
        T_red=1e4,
        backbone_net=FNN(layers=[1, 100, 100, 100, 100, 1]),
        train_data_size=500,
        test_data_size=501,
        sample_mode='uniform',
        GL_degree=100,
        prop:ArcPropSpline=None,
    ):
        self.R = R  # Arc radius [m]
        self.I = I  # Arc current [A]
        self.T_red = T_red  # Temperature normalization constant [K]
        self.Tb = Tb  # Boundary temperature [K]
        self.train_data_size = train_data_size  # Number of training collocation points
        self.test_data_size = test_data_size    # Number of test points
        self.sample_mode = sample_mode          # Sampling strategy
        self.GL_degree = GL_degree              # Gauss-Legendre quadrature degree
        # Compute Gauss-Legendre quadrature points and weights for arc conductance
        self.Xq, self.Wq = calc_GL_coefs(GL_degree)
        self.prop = prop  # Material properties (σ, κ, ε_nec)

        # Define 1D spatial domain [0, 1] (normalized radius)
        self.geo = Geo1D([0.0, 1.0])
        # Wrap backbone network with meta-learning specific architecture
        network = MetaStaArc1DNet(backbone_net)
        super().__init__(network)

        # Use smooth L1 loss (Huber loss) for robustness to outliers
        self.set_loss_func(F.smooth_l1_loss)

    
    def _define_loss_terms(self):
        """
        Define physics-informed loss terms for 1D stationary arc equation.

        Sets up two loss terms:
        1. Domain loss: PDE residual in the interior (0 < r < R)
        2. Boundary loss: Symmetry condition at centerline (r = 0)
        """
        def _pde_residual(network, x):
            """
            Compute PDE residual for the 1D stationary arc energy equation.

            The energy balance in cylindrical coordinates is enforced. The electric
            field E is computed from arc conductance using Gauss-Legendre quadrature.

            Parameters
            ----------
            network : nn.Module
                Meta-learning network (MetaStaArc1DNet).
            x : torch.Tensor
                Normalized radial coordinates, shape [N, 1].

            Returns
            -------
            torch.Tensor
                PDE residual, shape [N, 1] (should be zero for exact solution).
            """
            # Apply boundary condition transformation: T(r) = N(r)*(1-r) + Tb
            # This ensures T(R=1) = Tb automatically
            T = network(x)*(1.0 - x) + self.Tb/self.T_red
            
            # Compute temperature-dependent material properties
            kappa = self.prop.kappa(T.view(-1)*self.T_red).view(-1,1)  # Thermal conductivity
            sigma = self.prop.sigma(T.view(-1)*self.T_red).view(-1,1)  # Electrical conductivity
            nec = self.prop.nec(T.view(-1)*self.T_red).view(-1,1)      # Net emission coefficient

            # Compute arc conductance using Gauss-Legendre quadrature
            # G = πR² ∫₀¹ r·σ(T(r)) dr  (integral in normalized coordinates)
            Tq = network(self.Xq)*(1.0 - self.Xq) + self.Tb/self.T_red
            sigma_q = self.prop.sigma(Tq.view(-1)*self.T_red).view(-1,1)
            arc_cond = np.pi*self.R*self.R*torch.sum(self.Wq*self.Xq*sigma_q)

            # Compute energy source terms
            joule = sigma*(self.I/arc_cond)**2  # Joule heating: σ·E²
            radiation = 4*np.pi*nec             # Radiation loss: 4π·ε_nec
            net_energy = (joule - radiation)/self.T_red*self.R*self.R  # Normalized net energy

            # Compute thermal conduction term: (1/r) d/dr(r·κ·dT/dr)
            T_x = df_dX(T, x)           # First derivative: dT/dr
            T_term = x*kappa*T_x        # r·κ·dT/dr
            T_xx = df_dX(T_term, x)     # d/dr(r·κ·dT/dr)

            # PDE residual: (1/r)·d/dr(r·κ·dT/dr) + net_energy = 0
            # Multiply by r to avoid singularity: d/dr(r·κ·dT/dr) + r·net_energy = 0
            func = T_xx + x*net_energy
            return func
        
        def _bc_residual(network, x):
            """
            Compute boundary condition residual at centerline (r = 0).

            Enforces the symmetry condition: dT/dr|_{r=0} = 0.
            This is a natural consequence of cylindrical symmetry: the temperature
            gradient must vanish at the axis to ensure a unique, smooth solution.

            Parameters
            ----------
            network : nn.Module
                Meta-learning network.
            x : torch.Tensor
                Boundary point at r = 0, shape [1, 1].

            Returns
            -------
            torch.Tensor
                Boundary residual (should be zero), shape [1, 1].
            """
            # Compute temperature with boundary transformation
            T = network(x)*(1.0 - x) + self.Tb/self.T_red
            # Compute temperature gradient
            T_x = df_dX(T, x)
            # Residual: dT/dr should be zero at r=0
            return T_x
        
        # Sample collocation points in the domain (0 < r < 1)
        x_domain = self.geo.sample_domain(self.train_data_size, mode=self.sample_mode)
        
        # Sample boundary point at centerline (r = 0)
        x_bc = self.geo.sample_boundary()
        x_bc_left = x_bc[0]  # Left boundary corresponds to r=0
        
        # Add equation terms with weights
        # Domain: PDE residual in interior
        self.add_equation('Domain', _pde_residual, weight=1.0, data=x_domain)
        # Boundary: Symmetry condition at centerline (weighted higher for emphasis)
        self.add_equation('Boundary', _bc_residual, weight=10.0, data=x_bc_left)

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Make temperature predictions using the trained/adapted network.

        This method applies the trained meta-network (after adaptation) to predict
        temperature distributions at given radial locations. The boundary condition
        transformation is applied to ensure T(R) = Tb.

        Parameters
        ----------
        input_data : torch.Tensor
            Normalized radial coordinates (0 to 1), shape [N, 1].

        Returns
        -------
        torch.Tensor
            Normalized temperature predictions, shape [N, 1].
            Physical temperature: T_physical = output * self.T_red
        """
        self.network.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient computation for efficiency
            # Apply boundary transformation: T(r) = N(r)·(1-r) + Tb
            output = self.network(input_data)*(1.0 - input_data) + self.Tb/self.T_red
        return output


class StaArc1DTask(PINNTask):
    """
    Task wrapper for 1D stationary arc problem in meta-learning framework.

    This class encapsulates a specific arc discharge configuration as a meta-learning
    task. It creates the underlying PINN model, samples support and query sets, and
    provides the interface required by MetaPINN for meta-training and meta-testing.

    Task Definition
    ---------------
    Each task corresponds to a unique arc discharge problem instance:
    - Physical parameters: Arc radius R, current I, boundary temp Tb
    - Material properties: Plasma gas type (SF6, Ar, etc.)
    - Support set: Small set of collocation points for adaptation
    - Query set: Separate set for meta-validation

    Parameters
    ----------
    task_id : str
        Unique identifier for this arc configuration (e.g., 'Arc_I=200A_R=10mm').
    R : float
        Arc radius in meters.
    I : float
        Arc current in amperes.
    Tb : float, default=300.0
        Boundary temperature at r=R in Kelvin.
    T_red : float, default=1e4
        Temperature normalization constant.
    backbone_net : nn.Module, default=FNN([1,100,100,100,100,1])
        Backbone neural network architecture.
    thermo_file : str
        Path to thermodynamic properties file (κ, Cp, ρ vs T).
    nec_file : str
        Path to net emission coefficient file (ε_nec vs T).
    support_data_size : int, default=500
        Number of collocation points in support set.
    query_data_size : int, default=400
        Number of collocation points in query set.
    sample_mode : str, default='uniform'
        Sampling strategy ('uniform', 'random', 'lhs').

    Attributes
    ----------
    R : float
        Arc radius (stored for reference).
    pinn_model : MetaStaArc1DModel
        Underlying PINN model for this task.
    support_data : Dict[str, torch.Tensor]
        Support set collocation points {'Domain': x_spt, 'Boundary': x_bc}.
    query_data : Dict[str, torch.Tensor]
        Query set collocation points {'Domain': x_qry, 'Boundary': x_bc}.
    """
    def __init__(
        self,
        task_id: str,
        R: float,
        I: float,
        Tb: float = 300.0,
        T_red: float = 1e4,
        backbone_net: nn.Module = FNN(layers=[1, 100, 100, 100, 100, 1]),
        thermo_file: str = None,
        nec_file: str = None,
        support_data_size: int = 500,
        query_data_size: int = 400,
        sample_mode: str = 'uniform'
    ):
        """
        Initialize a stationary arc task for meta-learning.

        This constructor creates a complete task instance by:
        1. Setting up material properties from data files
        2. Creating the underlying MetaStaArc1DModel
        3. Sampling support and query sets (disjoint)
        4. Registering with parent PINNTask class

        The support and query sets are sampled independently to ensure they are
        disjoint, which is crucial for unbiased meta-learning.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task (e.g., 'Arc_I=200A_R=10mm').
        R : float
            Arc radius in meters.
        I : float
            Arc current in amperes.
        Tb : float, default=300.0
            Boundary temperature at r=R in Kelvin.
        T_red : float, default=1e4
            Temperature normalization constant (typically 10000K).
        backbone_net : nn.Module, default=FNN([1,100,100,100,100,1])
            Neural network architecture for temperature prediction.
        thermo_file : str, optional
            Path to thermodynamic properties CSV file (κ, Cp, ρ vs T).
        nec_file : str, optional
            Path to net emission coefficient CSV file (ε_nec vs T).
        support_data_size : int, default=500
            Number of collocation points in support set (inner loop).
        query_data_size : int, default=400
            Number of collocation points in query set (outer loop).
        sample_mode : str, default='uniform'
            Sampling strategy: 'uniform', 'random', or 'lhs'.

        Raises
        ------
        FileNotFoundError
            If thermo_file or nec_file is not found.
        """
        # Store arc radius for reference
        self.R = R
        
        # Load material properties from data files
        prop = ArcPropSpline(thermo_file, nec_file, R)
        
        # Create underlying PINN model for this arc configuration
        pinn_model = MetaStaArc1DModel(
            R=R, 
            I=I, 
            Tb=Tb, 
            T_red=T_red, 
            backbone_net=backbone_net, 
            train_data_size=support_data_size,  # Used for model setup
            test_data_size=query_data_size,     # Used for evaluation grid
            sample_mode=sample_mode, 
            prop=prop
        )

        # Sample support set for inner loop adaptation
        # Support set: small dataset for fast task-specific fine-tuning
        support_data = {
            'Domain': pinn_model.geo.sample_domain(support_data_size, mode=sample_mode),
            'Boundary': pinn_model.geo.sample_boundary()[0]
        }
        
        # Sample query set for meta-update (outer loop)
        # Query set: independent dataset for computing meta-gradients
        query_data = {
            'Domain': pinn_model.geo.sample_domain(query_data_size, mode=sample_mode),
            'Boundary': pinn_model.geo.sample_boundary()[0]
        }

        # Register with parent PINNTask class
        super().__init__(task_id, pinn_model, support_data, query_data)


