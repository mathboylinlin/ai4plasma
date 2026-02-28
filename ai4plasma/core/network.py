"""Core neural network architectures for scientific computing and physics-informed learning.

This module provides flexible, highly configurable neural network building blocks optimized
for physics-informed machine learning and operator learning problems in the AI4Plasma framework.

Network Classes
---------------
- `Network`: Abstract base class for all neural network architectures.
- `FNN`: Fully Connected Neural Network.
- `CNN`: Convolutional Neural Network.
- `RelaxLayer`: Relaxed hidden layer for NAS-PINN.
- `RelaxFNN`: Relaxed fully connected network for NAS-PINN.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ai4plasma.config import REAL, DEVICE


class Network(nn.Module, ABC):
    """Abstract base class for neural network architectures.
    
    This class serves as a base for all neural network implementations. All
    subclasses should override the forward() and init_weights() methods to
    implement specific architectures and initialization strategies.
    
    Attributes
    ----------
    None
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def init_weights(self):
        """Initialize network weights.
        
        This method should be overridden by subclasses to provide specific
        weight initialization strategies appropriate for the network architecture.

        Notes
        -----
        Abstract method that must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the network.
        
        Returns
        -------
        torch.Tensor
            Output tensor from the network.

        Notes
        -----
        Abstract method that must be implemented by subclasses.
        """
        pass


class FNN(Network):
    """Fully Connected Neural Network (Multi-layer Perceptron).
    
    A flexible multi-layer fully connected neural network with customizable depths,
    widths, activation functions, and weight initialization strategies. Suitable
    for function approximation and physics-informed machine learning tasks.
    
    Attributes
    ----------
    layers : list of int
        Number of neurons in each layer.
    act_fun : torch.nn.Module
        Activation function applied between layers.
    net : torch.nn.Sequential
        The complete network model.
    """

    def __init__(self, layers, act_fun=nn.Tanh(), use_BN=False, init_method='xavier') -> None:
        """Initialize the FNN.
        
        Parameters
        ----------
        layers : list of int
            Number of neurons in each layer. The first and last values are
            input and output dimensions, respectively.
        act_fun : torch.nn.Module, optional
            Activation function applied between layers. Default is Tanh().
        use_BN : bool, optional
            Whether to use batch normalization after each linear layer
            (except the output layer). Default is False.
        init_method : {'xavier', 'zero'}, optional
            Weight initialization method. 'xavier' uses Xavier/Glorot initialization,
            'zero' initializes all weights to zero. Default is 'xavier'.
        """
        super().__init__()
        self.layers = layers
        self.act_fun = act_fun
        self.net = self.linear_model(layers, act_fun, use_BN)
        self.init_weights(self.net, init_method)

    def linear_model(self, layers, activation, use_BN=False):
        """Construct a sequential multi-layer network.
        
        Parameters
        ----------
        layers : list of int
            Number of neurons in each layer.
        activation : torch.nn.Module
            Activation function to apply between layers.
        use_BN : bool, optional
            Whether to use batch normalization. Default is False.
        
        Returns
        -------
        torch.nn.Sequential
            Sequential model containing linear layers with optional batch norm
            and activation functions.
        """
        model = nn.Sequential()
        for i in range(0, len(layers) - 2):
            layer_name = 'linear%d' % (i + 1)
            activation_name = 'activation%d' % (i + 1)
            model.add_module(layer_name, nn.Linear(layers[i], layers[i + 1], dtype=REAL('torch')))
            
            if use_BN:
                model.add_module('BN%d' % (i + 1), nn.BatchNorm1d(layers[i + 1], dtype=REAL('torch')))
            model.add_module(activation_name, activation)

        layer_name = 'linear%d' % (len(layers) - 1)
        model.add_module(layer_name, nn.Linear(layers[-2], layers[-1], dtype=REAL('torch')))

        return model

    def init_weights(self, net, method='xavier'):
        """Initialize network weights using specified method.
        
        Parameters
        ----------
        net : torch.nn.Sequential
            The network whose weights to initialize.
        method : {'xavier', 'zero'}, optional
            Initialization method. Default is 'xavier'.
        """
        for m in net.modules():
            if isinstance(m, nn.Linear):
                if method == 'zero':
                    nn.init.constant_(m.weight, 0.0)
                else:  # default: 'xavier'
                    nn.init.xavier_normal_(m.weight)
                
                # bias
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Forward pass through the FNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        out = self.net(x)
        return out


class CNN(Network):
    """Convolutional Neural Network for scientific computing applications.
    
    A flexible convolutional neural network designed for physics-informed machine
    learning and operator learning in plasma physics and other scientific domains.
    Supports 1D, 2D, and 3D convolutions with customizable architecture parameters,
    batch normalization, pooling strategies, and an optional fully connected head.
    
    Attributes
    ----------
    conv_layers : list of int
        Channel counts for convolutional layers.
    fc_layers_config : list of int, optional
        Configured layer sizes for the fully connected head.
    input_dim : int
        Spatial dimension of input data (1, 2, or 3).
    act_fun : torch.nn.Module
        Activation function applied after conv and fc layers.
    use_BN : bool
        Whether batch normalization is used.
    use_pooling : bool
        Whether pooling layers are used.
    conv_net : torch.nn.Sequential
        Sequential container of convolutional layers.
    fc_net : torch.nn.Sequential, optional
        Sequential container of fully connected layers (lazily initialized).
    global_pool : torch.nn.Module, optional
        Global pooling layer (used if fc_layers is None).
    """
    
    def __init__(self, conv_layers, fc_layers=None, input_dim=2, act_fun=nn.ReLU(), 
                 use_BN=False, use_pooling=True, pooling_type='max', 
                 kernel_size=3, stride=1, padding=1, 
                 pooling_kernel_size=2, pooling_stride=None, pooling_padding=0,
                 init_method='xavier') -> None:
        """Initialize the CNN.
        
        Parameters
        ----------
        conv_layers : list of int
            Channel counts for convolutional layers. The first value must match
            the input channels.
        fc_layers : list of int, optional
            Neuron counts for fully connected layers. If None, global average
            pooling is used. If provided, the first value is automatically
            adjusted to match the flattened conv feature size. Default is None.
        input_dim : int, optional
            Spatial dimension of input data (1, 2, or 3). Default is 2.
        act_fun : torch.nn.Module, optional
            Activation function. Default is ReLU().
        use_BN : bool, optional
            Whether to use batch normalization. Default is False.
        use_pooling : bool, optional
            Whether to use pooling layers. Default is True.
        pooling_type : {'max', 'avg'}, optional
            Type of pooling to use. Default is 'max'.
        kernel_size : int or tuple, optional
            Convolution kernel size. Default is 3.
        stride : int or tuple, optional
            Convolution stride. Default is 1.
        padding : int or tuple, optional
            Convolution padding. Default is 1.
        pooling_kernel_size : int or tuple, optional
            Pooling kernel size. Default is 2.
        pooling_stride : int or tuple, optional
            Pooling stride. If None, defaults to pooling_kernel_size. Default is None.
        pooling_padding : int or tuple, optional
            Pooling padding. Default is 0.
        init_method : {'xavier', 'kaiming', 'zero'}, optional
            Weight initialization method. Default is 'xavier'.
        
        Raises
        ------
        ValueError
            If input_dim is not 1, 2, or 3.
        """
        super().__init__()
        
        # Validate input dimensions
        if input_dim not in [1, 2, 3]:
            raise ValueError("input_dim must be 1, 2, or 3")
        
        self.conv_layers = conv_layers
        self.fc_layers_config = fc_layers  # Store original config
        self.input_dim = input_dim
        self.act_fun = act_fun
        self.use_BN = use_BN
        self.use_pooling = use_pooling
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride if pooling_stride is not None else pooling_kernel_size
        self.pooling_padding = pooling_padding
        self.init_method = init_method
        
        # Build convolutional backbone
        self.conv_net = self._build_conv_layers(conv_layers, kernel_size, stride, padding,
                                                pooling_kernel_size, self.pooling_stride, pooling_padding)
        
        # FC net will be built lazily on first forward pass if fc_layers is specified
        self.fc_net = None
        self.fc_layers = None
        self._fc_initialized = False
        
        if fc_layers is None:
            # Use global average pooling for regression tasks
            self.global_pool = self._get_global_pool()
            self.use_fc = False
        else:
            self.use_fc = True
        
        # Initialize conv weights
        self.init_weights(init_method)
    
    def _get_conv_layer(self):
        """Get the appropriate convolution layer class.
        
        Returns
        -------
        type
            Convolution layer class (Conv1d, Conv2d, or Conv3d) based on input_dim.
        """
        if self.input_dim == 1:
            return nn.Conv1d
        elif self.input_dim == 2:
            return nn.Conv2d
        else:  # input_dim == 3
            return nn.Conv3d
    
    def _get_bn_layer(self, channels):
        """Get the appropriate batch normalization layer.
        
        Parameters
        ----------
        channels : int
            Number of channels for batch normalization.
        
        Returns
        -------
        torch.nn.Module
            BatchNorm1d, BatchNorm2d, or BatchNorm3d layer.
        """
        if self.input_dim == 1:
            return nn.BatchNorm1d(channels, dtype=REAL('torch'))
        elif self.input_dim == 2:
            return nn.BatchNorm2d(channels, dtype=REAL('torch'))
        else:  # input_dim == 3
            return nn.BatchNorm3d(channels, dtype=REAL('torch'))
    
    def _get_pool_layer(self, kernel_size=2, stride=None, padding=0):
        """Get the appropriate pooling layer.
        
        Parameters
        ----------
        kernel_size : int or tuple, optional
            Kernel size for pooling. Default is 2.
        stride : int or tuple, optional
            Stride for pooling. If None, defaults to kernel_size. Default is None.
        padding : int or tuple, optional
            Padding for pooling. Default is 0.
        
        Returns
        -------
        torch.nn.Module
            MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, or AvgPool3d layer.
        """
        if stride is None:
            stride = kernel_size
            
        if self.pooling_type == 'max':
            if self.input_dim == 1:
                return nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
            elif self.input_dim == 2:
                return nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
            else:  # input_dim == 3
                return nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
        else:  # avg pooling
            if self.input_dim == 1:
                return nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
            elif self.input_dim == 2:
                return nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
            else:  # input_dim == 3:
                return nn.AvgPool3d(kernel_size, stride=stride, padding=padding)
    
    def _get_global_pool(self):
        """Get global average pooling layer.
        
        Returns
        -------
        torch.nn.Module
            AdaptiveAvgPool1d, AdaptiveAvgPool2d, or AdaptiveAvgPool3d layer.
        """
        if self.input_dim == 1:
            return nn.AdaptiveAvgPool1d(1)
        elif self.input_dim == 2:
            return nn.AdaptiveAvgPool2d(1)
        else:  # input_dim == 3
            return nn.AdaptiveAvgPool3d(1)
    
    def _calculate_feature_size(self, input_tensor):
        """Calculate the flattened feature size after convolution.
        
        Performs a forward pass through the convolutional layers without
        recording gradients to determine the size of the flattened output.
        
        Parameters
        ----------
        input_tensor : torch.Tensor
            Actual input tensor to calculate feature size from.
        
        Returns
        -------
        int
            The flattened feature size after convolution and pooling.
        """
        with torch.no_grad():
            out = self.conv_net(input_tensor)
            flattened_size = out.view(out.size(0), -1).size(1)
        
        return flattened_size
    
    def _build_conv_layers(self, layers, kernel_size, stride, padding,
                          pooling_kernel_size, pooling_stride, pooling_padding):
        """Build the convolutional layers of the network.
        
        Parameters
        ----------
        layers : list of int
            Channel numbers for each convolutional layer.
        kernel_size : int or tuple
            Kernel size for convolutions.
        stride : int or tuple
            Stride for convolutions.
        padding : int or tuple
            Padding for convolutions.
        pooling_kernel_size : int or tuple
            Kernel size for pooling.
        pooling_stride : int or tuple
            Stride for pooling.
        pooling_padding : int or tuple
            Padding for pooling.
        
        Returns
        -------
        torch.nn.Sequential
            Sequential container of convolutional layers with optional batch
            normalization, activation, and pooling.
        """
        conv_layer_fn = self._get_conv_layer()
        model = nn.Sequential()
        
        for i in range(len(layers) - 1):
            # Convolutional layer
            conv_name = f'conv{i + 1}'
            model.add_module(conv_name, conv_layer_fn(
                layers[i], layers[i + 1], 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                dtype=REAL('torch')
            ))
            
            # Batch normalization (optional)
            if self.use_BN:
                bn_name = f'bn{i + 1}'
                model.add_module(bn_name, self._get_bn_layer(layers[i + 1]))
            
            # Activation function
            act_name = f'activation{i + 1}'
            model.add_module(act_name, self.act_fun)
            
            # Pooling layer (optional)
            if self.use_pooling:
                pool_name = f'pool{i + 1}'
                model.add_module(pool_name, self._get_pool_layer(
                    kernel_size=pooling_kernel_size,
                    stride=pooling_stride,
                    padding=pooling_padding
                ))
        
        return model
    
    def _build_fc_layers(self, layers):
        """Build the fully connected layers of the network.
        
        Parameters
        ----------
        layers : list of int
            Neuron counts for each fully connected layer.
        
        Returns
        -------
        torch.nn.Sequential
            Sequential container of fully connected layers with activation
            functions (except on the output layer).
        """
        model = nn.Sequential()
        
        for i in range(len(layers) - 1):
            # Linear layer
            linear_name = f'fc{i + 1}'
            model.add_module(linear_name, nn.Linear(layers[i], layers[i + 1], dtype=REAL('torch')))
            
            # Activation (except for the last layer)
            if i < len(layers) - 2:
                act_name = f'fc_activation{i + 1}'
                model.add_module(act_name, self.act_fun)
        
        return model
    
    def init_weights(self, method='xavier'):
        """Initialize network weights using the specified method.
        
        Applies the initialization strategy to all convolutional, linear,
        and batch normalization layers in the network.
        
        Parameters
        ----------
        method : {'xavier', 'kaiming', 'zero'}, optional
            Weight initialization method. Default is 'xavier'.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                if method == 'zero':
                    nn.init.constant_(m.weight, 0.0)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:  # default: 'xavier'
                    nn.init.xavier_normal_(m.weight)
                
                # Initialize bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """Forward pass through the CNN with lazy FC layer initialization.
        
        On the first forward pass with fc_layers, automatically adjusts the
        fully connected layer input size based on the actual conv output shape.
        Subsequent forwards use the pre-initialized FC layers.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape depending on input_dim:
            
            - 1D: (batch_size, channels, length)
            - 2D: (batch_size, channels, height, width)
            - 3D: (batch_size, channels, depth, height, width)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_features).
        """
        # Lazily initialize FC layers on first forward pass
        if self.use_fc and not self._fc_initialized:
            # Calculate actual feature size from input
            actual_feature_size = self._calculate_feature_size(x)
            
            # Adjust fc_layers config if needed
            if self.fc_layers_config[0] != actual_feature_size:
                adjusted_fc_layers = [actual_feature_size] + self.fc_layers_config[1:]
                print(f"[INFO] CNN: Automatically adjusted fc_layers[0] from {self.fc_layers_config[0]} to {actual_feature_size} (based on actual input shape {x.shape})")
                self.fc_layers = adjusted_fc_layers
            else:
                self.fc_layers = self.fc_layers_config
            
            # Build FC layers
            self.fc_net = self._build_fc_layers(self.fc_layers)
            self.fc_net = self.fc_net.to(x.device)
            self._fc_initialized = True
        
        # Pass through convolutional layers
        out = self.conv_net(x)
        
        # Process through FC layers or global pooling
        if self.use_fc:
            # Flatten for fully connected layers
            out = out.view(out.size(0), -1)  # (batch_size, flattened_features)
            out = self.fc_net(out)
        else:
            # Global average pooling for regression
            out = self.global_pool(out)
            out = out.view(out.size(0), -1)  # Flatten to (batch_size, channels)
        
        return out
    
    def get_feature_size(self, input_shape):
        """Calculate the output feature size after convolutional layers.
        
        Used to determine the input size required for the first fully connected
        layer when designing network architectures manually.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input tensor (channels, spatial_dims...)
        
        Returns
        -------
        int
            Number of features after convolution and pooling operations.
        """
        # Create a dummy input tensor
        if self.input_dim == 1:
            dummy_input = torch.zeros(1, *input_shape, dtype=REAL('torch'))
        elif self.input_dim == 2:
            dummy_input = torch.zeros(1, *input_shape, dtype=REAL('torch'))
        else:  # input_dim == 3
            dummy_input = torch.zeros(1, *input_shape, dtype=REAL('torch'))
        
        # Pass through conv layers
        with torch.no_grad():
            out = self.conv_net(dummy_input)
            return out.view(1, -1).size(1)
        

class RelaxLayer(Network):
    """Relaxed hidden layer for NAS-PINN architecture search.
    
    Implements a relaxed fully connected hidden layer where the effective
    operation is determined by learnable architecture parameters ``g``. The
    layer blends identity and nonlinear transformations with soft selections
    over candidate neuron counts.
    
    Parameters
    ----------
    C_in : int
        Input feature dimension.
    neuron_list : list of int
        Candidate neuron counts. ``neuron_list[0]`` is treated as identity.
    act_fun : torch.nn.Module, optional
        Activation function applied after the linear operator. Default is Tanh.
    init_method : {'xavier', 'zero'}, optional
        Weight initialization method for the linear operator. Default is 'xavier'.
    
    Attributes
    ----------
    C_in : int
        Input feature dimension.
    neuron_list : list of int
        Candidate neuron counts (including identity as index 0).
    op : torch.nn.Sequential
        Linear + activation operator for the maximal neuron count.
    masks : torch.Tensor
        Binary masks used to select subsets of neurons.
    """
    def __init__(self, C_in, neuron_list, act_fun=nn.Tanh(), init_method='xavier'):
        """Initialize a relaxed hidden layer.
        
        Parameters
        ----------
        C_in : int
            Input feature dimension.
        neuron_list : list of int
            Candidate neuron counts. The last value is the maximum width.
        act_fun : torch.nn.Module, optional
            Activation function applied after the linear operator. Default is Tanh.
        init_method : {'xavier', 'zero'}, optional
            Weight initialization method for the linear operator. Default is 'xavier'.
        """
        super().__init__()
        self.C_in = C_in
        self.neuron_list = neuron_list  # neuron_list[0] is always 0, representing Identity operation
        self.op = nn.Sequential(
            nn.Linear(C_in, neuron_list[-1]),
            act_fun)
        i = 0
        for neuron in neuron_list[1:]:
            one = torch.ones(1,int(neuron))              
            if neuron < neuron_list[-1]:
                zero = torch.zeros(1,int(neuron_list[-1] - neuron))
                mask = torch.cat([one,zero], 1)
            else:
                mask = one
            if i < 1:
                self.masks = mask
                i += 1
            else:
                self.masks = torch.cat((self.masks, mask), 0)
                i += 1
        self.masks = self.masks.to(DEVICE())

        self.init_weights(self.op, init_method)
    
    def forward(self, x, g):
        """Forward pass with relaxed architecture parameters.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, C_in).
        g : torch.Tensor
            Architecture parameter tensor for this layer. The first two entries
            control identity vs. nonlinear mixing; the remaining entries select
            neuron counts.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, max(neuron_list)).
        """
        # double-level: g[0] is a, g[1] is b, b contains all neurons
        g = g.view(1, -1)
        ab = F.softmax(g[:,:2], dim=-1)
        a = ab[0,0]
        b = ab[0,1]
        g_neuron = F.softmax(g[:,2:], dim=-1)

        patch = torch.zeros(x.size(0), self.neuron_list[-1] - self.C_in)
        patch = patch.to(DEVICE())
        identity = torch.cat((x, patch), 1)
        temp = self.op(x)
        weight = (torch.mm(g_neuron, self.masks)).T
        temp = temp * weight[:,0] * b
        out = temp + a * identity       

        return out
    
    def init_weights(self, net, method='xavier'):
        """Initialize weights for the linear operator.
        
        Parameters
        ----------
        net : torch.nn.Sequential
            Operator whose weights will be initialized.
        method : {'xavier', 'zero'}, optional
            Weight initialization method. Default is 'xavier'.
        """
        for m in net.modules():
            if isinstance(m, nn.Linear):
                if method == 'zero':
                    nn.init.constant_(m.weight, 0.0)
                else:  # default: 'xavier'
                    nn.init.xavier_normal_(m.weight)
                
                # bias
                nn.init.constant_(m.bias, 0.0)
    

class RelaxFNN(Network):
    """Relaxed fully connected network for NAS-PINN.
    
    Builds a stack of relaxed layers whose architectures are controlled by
    learnable parameters. The network supports architecture search by learning
    soft selections over identity vs. nonlinear paths and neuron counts.
    
    Parameters
    ----------
    layers : int
        Number of relaxed layers.
    C_in_list : list of int
        Input dimension for each layer (length equals ``layers``).
    neuron_list : list of int
        Candidate neuron counts for each relaxed layer.
    
    Attributes
    ----------
    layers : int
        Number of relaxed layers.
    C_in_list : list of int
        Input dimensions for each layer.
    neuron_list : list of int
        Candidate neuron counts.
    network : torch.nn.ModuleList
        Stack of RelaxLayer modules.
    gs : torch.Tensor
        Architecture parameters with shape (layers, len(neuron_list) + 1).
    """
    def __init__(self, layers, C_in_list, neuron_list):
        """Initialize the relaxed FNN.
        
        Parameters
        ----------
        layers : int
            Number of relaxed layers.
        C_in_list : list of int
            Input dimension for each layer.
        neuron_list : list of int
            Candidate neuron counts for each layer.
        """
        super().__init__()
        self.layers = layers
        self.C_in_list = C_in_list
        self.neuron_list = neuron_list

        self.output_layer = nn.Linear(C_in_list[-1], 1)
        self.init_weights()
        self.build_up()

    def init_weights(self):
        """Initialize architecture parameters ``gs``.
        
        Creates learnable parameters for each layer, with the first two entries
        controlling identity vs. nonlinear mixing and the remaining entries
        selecting neuron counts.
        """
        self.gs = Variable(1e-3 * torch.randn(self.layers, len(self.neuron_list)+1).to(DEVICE()), requires_grad=True)
        
        self.arch_para = [self.gs]
    
    def load_gs(self, arch_param):
        """Load architecture parameters from an external list.
        
        Parameters
        ----------
        arch_param : list of torch.Tensor
            List where the first element contains the architecture parameters.
        """
        self.gs = arch_param[0]
        self.arch_para = [self.gs]

    def arch_parameters(self):
        """Return architecture parameters for optimization.
        
        Returns
        -------
        list of torch.Tensor
            List containing the architecture parameter tensor.
        """
        self.arch_para = [self.gs]
        return self.arch_para

    def build_up(self):
        """Build the relaxed network stack.
        
        Creates a ModuleList of RelaxLayer instances based on ``C_in_list`` and
        ``neuron_list``.
        """
        self.network = nn.ModuleList()

        for i in range(self.layers):
            self.network.append(RelaxLayer(self.C_in_list[i], self.neuron_list))

    def searched_neuron(self, threshold=1e-3):
        """Derive the discrete architecture from learned parameters.
        
        If the identity vs. nonlinear weights are close, retain a residual
        connection and append the selected neuron count. Otherwise, select the
        dominant option.

        Parameters
        ----------
        threshold : float, optional
            Threshold for determining if identity and nonlinear paths are close. Default is 1e-3.
        
        Returns
        -------
        list
            List of selected neuron descriptors per layer.
        """
        final_neuron = []
        for i in range(self.layers):
            g = self.gs[i]
            ab = F.softmax(g[:2], dim=-1)
            if abs(ab[0] - ab[1]) > threshold:
                if torch.argmax(ab) == 0:
                    index = 0
                else:
                    g = F.softmax(g[2:], dim=-1)
                    index = torch.argmax(g)+1
                neuron = self.neuron_list[index]
            else:
                neuron = '0+'
                g = F.softmax(g[2:], dim=-1)
                index = torch.argmax(g)+1
                neuron += str(self.neuron_list[index])
            final_neuron.append(neuron)

        return final_neuron
    
    def forward(self, x):
        """Forward pass through the relaxed FNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, C_in_list[0]).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        for i in range(self.layers):
            x = self.network[i](x, self.gs[i])
        y = self.output_layer(x)
        return y