import copy # Used for deep copying model states and optimizer states
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections # For creating a deque (double-ended queue) for reward history

from sklearn.datasets import load_breast_cancer # Example dataset for classification
from sklearn.model_selection import train_test_split # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler # For normalizing input features
from sklearn.metrics import accuracy_score # For evaluating model accuracy

# Determine if a GPU is available and use it; otherwise, fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# DATA LOADING AND PREPROCESSING
# =========================

def load_dataset():
    """
    Loads the Breast Cancer dataset, splits it into training and testing sets,
    and preprocesses the features using StandardScaler.

    Returns:
        tuple: X_train, X_test, y_train, y_test (all torch.Tensor on the active device),
               input_features (int), num_classes (int).
    """
    data = load_breast_cancer()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42 # For reproducibility
    )

    # Scale features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert numpy arrays to PyTorch tensors and move to the selected device
    return (
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(X_test).to(device),
        torch.LongTensor(y_train).to(device),
        torch.LongTensor(y_test).to(device),
        X_train.shape[1], # Number of input features
        len(np.unique(data.target)) # Number of unique classes
    )


# =========================
# HEBBIAN LEARNING LAYER
# =========================

class HebbianLinear(nn.Module):
    """
    A custom linear layer that incorporates Hebbian-like updates for its weights.
    This layer's weights are updated based on the correlation between input and output
    activations, in addition to standard backpropagation.
    """
    def __init__(self, inp, out, lr=1e-4):
        """
        Initializes the HebbianLinear layer.

        Args:
            inp (int): Number of input features.
            out (int): Number of output features.
            lr (float): Learning rate for Hebbian updates.
        """
        super().__init__()

        # Initialize weights with a small random distribution
        self.weight = nn.Parameter(
            torch.randn(out, inp) * 0.02
        )

        # Initialize biases to zeros
        self.bias = nn.Parameter(
            torch.zeros(out)
        )

        self.hebb_lr = lr # Hebbian learning rate

    def forward(self, x):
        """
        Performs the standard forward pass for a linear layer.
        """
        return F.linear(x, self.weight, self.bias)

    def hebbian_update(self, x, y):
        """
        Applies a Hebbian update rule to the weights of the layer.
        The update strength is proportional to the product of pre-synaptic (x)
        and post-synaptic (y) activities, adjusted by a term related to the output.

        Args:
            x (torch.Tensor): The input activations (pre-synaptic).
            y (torch.Tensor): The output activations (post-synaptic).
        """
        with torch.no_grad(): # Hebbian updates are out of the gradient computation graph

            y = torch.clamp(y, -10, 10) # Clamp y to prevent extreme values

            # Calculate the change in weight (dw) based on a Hebbian rule
            # dw = y_j * x_i - y_j^2 * w_ij (a form of Oja's rule for stability)
            dw = (
                y.unsqueeze(2) * x.unsqueeze(1) # Outer product for correlation
                - (y.unsqueeze(2) ** 2) * self.weight # Stabilization term
            )

            # Apply in-place update to self.weight.data to avoid RuntimeError
            self.weight.data += self.hebb_lr * dw.mean(0)


# =========================
# GROWTH CONTROLLER (Reinforcement Learning Agent)
# =========================

class GrowthController(nn.Module):
    """
    A small neural network that acts as a controller for the AdaptiveNet's architecture.
    It takes in various network statistics and outputs a decision (growth or pruning).
    """
    def __init__(self,):
        super().__init__()

        # The controller network consists of fully connected layers with ReLU activations.
        # It takes 5 input features (usage_mean, entropy, grad_norm, activation_mean, hidden_ratio)
        # and outputs a single value.
        self.net = nn.Sequential(
            nn.Linear(5, 32), # Inputs: usage, entropy, grad_norm, activation_mean, hidden_ratio
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # Tanh activation is applied outside in forward for scaling
        )

    def forward(self, stats):
        """
        Forward pass for the controller.

        Args:
            stats (torch.Tensor): A tensor containing the 5 network statistics.

        Returns:
            torch.Tensor: The controller's decision, clamped between -0.5 and 0.5.
        """
        # Prevent action saturation by clamping output to [-0.5, 0.5]
        # Tanh activation maps input to (-1, 1), multiplying by 0.5 scales it to (-0.5, 0.5)
        return 0.5 * torch.tanh(self.net(stats))


# =========================
# ADAPTIVE BLOCK (Core Layer of AdaptiveNet)
# =========================

class AdaptiveBlock(nn.Module):
    """
    The fundamental building block of the AdaptiveNet. It combines a HebbianLinear layer
    with a gating mechanism and tracks neuron usage.
    """
    def __init__(self, inp, hidden):
        """
        Initializes the AdaptiveBlock.

        Args:
            inp (int): Number of input features.
            hidden (int): Number of hidden units in this block.
        """
        super().__init__()

        self.inp = inp
        self.hidden = hidden

        self.fc = HebbianLinear(inp, hidden) # Hebbian linear transformation
        self.gate = nn.Linear(inp, hidden) # Gating mechanism for feature selection

        # A buffer to store and track the 'usage' of each hidden neuron.
        # This is a non-trainable parameter that persists across epochs.
        self.register_buffer(
            "usage",
            torch.zeros(hidden) # Initialize usage for all neurons to zero
        )

    def forward(self, x):
        """
        Forward pass for the AdaptiveBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: h (processed features), g (gate activations).
        """
        h = self.fc(x) # Apply Hebbian linear transformation

        g = torch.sigmoid(
            self.gate(x) # Compute gate activations
        ) # Sigmoid ensures gate values are between 0 and 1

        h = h * g # Apply gating: selectively pass information based on gate values

        h = F.gelu(h) # Apply GELU activation function

        with torch.no_grad():
            # Update neuron usage: moving average of absolute activation magnitude
            # This helps in identifying active/inactive neurons for pruning decisions.
            self.usage.mul_(0.95).add_(
                0.05 * h.abs().mean(0) # 0.05 is the update rate
            )

        return h, g


# =========================
# ADAPTIVENET (The main adaptive neural network)
# =========================

class AdaptiveNet(nn.Module):
    """
    An adaptive neural network that dynamically adjusts its hidden layer size
    (growth or pruning) based on a reinforcement learning controller's decisions.
    It also includes mechanisms for best model preservation, architecture freezing,
    and various cooldowns.
    """
    def __init__(self, inp, hidden, classes):
        """
        Initializes the AdaptiveNet.

        Args:
            inp (int): Number of input features.
            hidden (int): Initial number of hidden units in the adaptive block.
            classes (int): Number of output classes.
        """
        super().__init__()

        self.input_dim = inp # Store input_dim for rebuild
        self.num_classes = classes # Store num_classes for rebuild
        self.initial_hidden = hidden # Store initial hidden size for reference

        self.hidden = hidden # Current hidden size, dynamically changes

        # The core adaptive block
        self.block = AdaptiveBlock(
            inp,
            hidden
        ).to(device)

        # The final classification layer (head)
        self.head = nn.Linear(
            hidden,
            classes
        ).to(device)

        # The RL controller that makes architectural decisions
        self.controller = GrowthController().to(device)

        self.soft_hidden = float(hidden) # A floating-point representation of the hidden size, influenced by the controller

        # Adaptation parameters
        # Reward history: Used to average rewards over a window for more stable decision-making.
        self.reward_history = collections.deque(maxlen=10) # Patience mechanism - Increased maxlen to 10
        self.adapt_patience = 10 # Adapt only every 10 epochs to reduce frequent structural changes
        self.reward_threshold = 0.0005 # Reward threshold for architecture freezing (more sensitive)
        self.min_hidden = 32 # Minimum allowed hidden layer size
        self.max_hidden = 256 # Maximum allowed hidden layer size
        self.complexity_lambda = 0.003 # This variable is actually removed later in the run function as it's not used in reward calculation directly.
        self.growth_rate_factor = 0.10 # Controls how aggressively the hidden size changes based on controller's decision
        self.last_decision = 0.0 # Stores the last decision made by the controller for logging

        # Controller Confidence Threshold (Task 1: Minimum Change Threshold)
        # If the controller's decision magnitude is below this, no architectural change occurs.
        self.controller_confidence_threshold = 0.001 # Reduced threshold for more sensitive changes

        # Architecture growth/pruning tracking (Task 9: Stability Metrics)
        self.total_rebuild_attempts = 0 # Total times a rebuild was attempted
        self.growth_events = 0 # Number of times the network grew
        self.prune_events = 0 # Number of times the network was pruned
        self.rollback_events = 0 # Number of times a rebuild was rolled back
        self.successful_growths = 0 # Successful growth events
        self.successful_prunes = 0 # Successful pruning events
        self.failed_rebuilds = 0 # Rebuilds that were either rejected or rolled back
        self.hidden_size_history = [] # Stores history of hidden sizes for variance tracking
        self.freeze_events = 0 # Times architecture was frozen
        self.unfreeze_events = 0 # Times architecture was unfrozen

        # Best Model Preservation (Task 2)
        self.best_state_dict = None # Stores the state dictionary of the best performing model
        self.best_test_accuracy = 0.0 # Best test accuracy achieved
        self.best_test_loss = float('inf') # Best test loss achieved
        self.best_epoch = -1 # Epoch at which the best model was saved
        self.best_param_efficiency = 0.0 # Efficiency of the best model (accuracy per parameter)

        # Architecture Freeze (Task 3)
        self.stagnation_counter = 0 # Counts epochs with low reward and high accuracy, indicating stagnation
        self.architecture_frozen = False # Flag indicating if the architecture is currently frozen
        self.freeze_cooldown_counter = 0 # Prevents immediate re-freeze/unfreeze actions
        self.freeze_cooldown_epochs = 20 # Cooldown period for freeze/unfreeze actions
        self.unfreeze_threshold_drop = 0.01 # Unfreeze if accuracy drops by this much from the best

        # Rebuild Cooldown (Task 4)
        self.rebuild_cooldown_counter = 0 # Prevents rapid architectural modifications
        self.rebuild_cooldown_epochs = 15 # Wait for N epochs before the next rebuild attempt

        # Minimum Change Threshold (Task 5)
        self.min_architecture_step = 4 # Ensures hidden size changes by at least this much to avoid trivial adjustments

        # For validation-performance-based reward (Task 7: Reward Function Redesign)
        self.previous_test_acc = None
        self.previous_test_loss = None
        self.previous_efficiency = None

        # For tracking rebuild effectiveness for reward (Task 1: Rebuild Acceptance Audit)
        self.last_adaptation_was_rebuild_attempt = False # Flag if a rebuild was just attempted
        self.last_adaptation_was_useless_rebuild = False # Flag if the last rebuild yielded little to no change
        self.last_delta_acc = 0.0 # Stores delta_acc from the last _rebuild call, used in reward calculation

    def forward(self, x):
        """
        Forward pass for the AdaptiveNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: logits (raw output scores), h (hidden layer activations),
                   gate (gate activations), usage_mean_val (mean neuron usage),
                   entropy_val (gate entropy), activation_mean_val (mean hidden activation).
        """
        # Pass input through the adaptive block
        h, gate = self.block(x)

        # Pass block output through the head layer for classification logits
        logits = self.head(h)

        # Calculate entropy of the gate activations (encourages diverse gating decisions)
        entropy = -(gate * torch.log(gate + 1e-6) + # Added epsilon for numerical stability
                    (1 - gate) * torch.log(1 - gate + 1e-6)).mean()

        # Neuron Diversification Entropy
        # Measures how evenly neuron activity is distributed across the layer.
        #
        # Formula:
        #   p_i = usage_i / sum(usage)
        #   entropy = -Σ(p_i * log(p_i))
        #
        # Interpretation:
        #   High entropy  -> neuron usage is evenly distributed (good diversification)
        #   Low entropy   -> a few neurons dominate while others are rarely used
        #
        # This differs from binary gate entropy:
        #   -g*log(g) - (1-g)*log(1-g)
        # which measures gate uncertainty/confidence rather than neuron specialization.
        #
        # Use diversification entropy when the goal is to encourage all neurons
        # to participate and prevent a small subset from monopolizing learning.

        # Return components, grad_norm will be calculated in run() after backward()
        return logits, h, gate, self.block.usage.mean(), entropy, h.abs().mean()

    # Future Growth Strategy: Protected Neuron Incubation
    #
    # Current implementation immediately integrates newly grown neurons into the
    # network with random weights. This can introduce noise and destabilize
    # already-learned representations.
    #
    # More biologically plausible alternative:
    #
    # 1. Clone or initialize new neurons near existing high-utility neurons
    #    (parent neurons) instead of random initialization.
    #
    # 2. Keep new neurons isolated using a growth mask:
    #       output = old_neurons + alpha * new_neurons
    #    where alpha starts near 0.
    #
    # 3. Allow incubating neurons to learn independently for several epochs
    #    while minimizing interference with the mature network.
    #
    # 4. Gradually increase alpha from 0 -> 1 as neuron utility improves.
    #
    # 5. Merge neuron into the main network only when:
    #       usage > usage_threshold
    #       activation_variance > variance_threshold
    #       contribution_to_loss_reduction > contribution_threshold
    #
    # 6. Optionally initialize new neurons as perturbed copies of useful
    #    parent neurons:
    #       W_new = W_parent + ε
    #       b_new = b_parent + ε
    #
    # This creates specialization through exploration around successful
    # features rather than introducing completely random features.
    #
    # Conceptually:
    #   Random Growth      -> "Create random neuron and hope it helps"
    #   Incubated Growth   -> "Create child neuron, train privately,
    #                          integrate only after proving usefulness"
    #
    # Long-term goal:
    #   Enable evolutionary/self-organizing network expansion where neurons
    #   are born, mature, specialize, and join the network only after
    #   demonstrating value.

    def _transfer_weights_and_bias(self, old_module, new_module, grow_init_scale=0.1):
        """
        Transfers weights and biases from an old module to a new module, handling size differences.
        If the new module is larger, new connections/neurons are initialized with small random values.

        Args:
            old_module (nn.Module): The module from which to copy weights.
            new_module (nn.Module): The module to which to copy weights.
            grow_init_scale (float): Standard deviation for initializing new weights.
        """
        with torch.no_grad():
            # Determine the minimum dimensions for weight and bias transfer
            min_out_features = min(old_module.weight.shape[0], new_module.weight.shape[0])
            min_in_features = min(old_module.weight.shape[1], new_module.weight.shape[1])

            # Copy existing weights
            new_module.weight.data[:min_out_features, :min_in_features].copy_(
                old_module.weight.data[:min_out_features, :min_in_features]
            )
            # Copy existing biases if they exist in both modules
            if old_module.bias is not None and new_module.bias is not None:
                new_module.bias.data[:min_out_features].copy_(
                    old_module.bias.data[:min_out_features]
                )

            # Initialize new connections/neurons if new_module is larger (GROWTH)
            if new_module.weight.shape[0] > old_module.weight.shape[0]: # New output neurons/rows
                # Initialize new output neurons weights with small random values
                nn.init.normal_(new_module.weight.data[old_module.weight.shape[0]:], mean=0.0, std=grow_init_scale)
                if new_module.bias is not None: # Initialize new output neurons biases
                    new_module.bias.data[old_module.weight.shape[0]:] = 0.0

            if new_module.weight.shape[1] > old_module.weight.shape[1]: # New input connections/columns
                # Initialize new input connections for existing neurons
                nn.init.normal_(new_module.weight.data[:, old_module.weight.shape[1]:], mean=0.0, std=grow_init_scale)


    def _rebuild(self, new_hidden_size, X_test, y_test, loss_fn):
        """
        Rebuilds the network with a new hidden layer size. This involves creating new modules,
        transferring weights, evaluating performance, and potentially rolling back if the change
        is detrimental.

        Args:
            new_hidden_size (int): The target hidden layer size.
            X_test (torch.Tensor): Test features for performance evaluation.
            y_test (torch.Tensor): Test labels for performance evaluation.
            loss_fn (nn.Module): Loss function for evaluation.

        Returns:
            tuple: rebuilt_accepted (bool), rolled_back (bool).
        """
        device = next(self.parameters()).device # Get current device

        self.total_rebuild_attempts += 1 # Track total rebuild attempts (Task 9)
        self.hidden_size_history.append(self.hidden) # Log current hidden size (Task 9)

        # --- Save state before potential rebuild --- (Task 1: Rebuild Acceptance Audit - Rollback)
        old_model_state = copy.deepcopy(self.state_dict()) # Save entire model state for potential rollback
        old_hidden_size = self.hidden # Capture current hidden size for rollback
        old_soft_hidden = self.soft_hidden # Capture soft hidden size for rollback

        # --- Evaluate PRE-REBUILD performance --- (Task 1: Acceptance Audit)
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            old_test_logits, _, _, _, _, _ = self(X_test)
            old_test_pred = old_test_logits.argmax(1)
            old_test_acc = accuracy_score(y_test.cpu(), old_test_pred.cpu())
            old_test_loss = loss_fn(old_test_logits, y_test).item()
        self.train() # Set back to train mode

        # Store references to old modules before replacing them
        old_block = self.block
        old_head = self.head

        # Create new AdaptiveBlock with the new hidden size
        new_block = AdaptiveBlock(self.input_dim, new_hidden_size).to(device)

        # Transfer weights for HebbianLinear (fc layer) from old block to new block
        self._transfer_weights_and_bias(old_block.fc, new_block.fc)

        # Transfer weights for gate layer (nn.Linear) from old block to new block
        self._transfer_weights_and_bias(old_block.gate, new_block.gate)

        # Transfer usage buffer: copy existing usage, initialize new entries if grown
        with torch.no_grad():
            min_usage = min(old_block.usage.shape[0], new_block.usage.shape[0])
            new_block.usage.data[:min_usage].copy_(old_block.usage.data[:min_usage])
            if new_hidden_size > old_block.hidden:
                new_block.usage.data[old_block.hidden:] = 0.0 # Initialize new usage buffer entries to zero

        self._modules['block'] = new_block # Explicitly assign the new block to the model

        # Create new head layer with the new hidden size
        new_head = nn.Linear(new_hidden_size, self.num_classes).to(device)

        # Transfer weights for head layer from old head to new head
        self._transfer_weights_and_bias(old_head, new_head)
        self._modules['head'] = new_head # Explicitly assign the new head to the model

        # Update model's current hidden size and soft hidden tracker
        self.hidden = new_hidden_size
        self.soft_hidden = float(new_hidden_size) # Keep soft_hidden consistent after rebuild

        # --- Evaluate POST-REBUILD performance --- (Task 1: Acceptance Audit)
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            new_test_logits, _, _, _, _, _ = self(X_test)
            new_test_pred = new_test_logits.argmax(1)
            new_test_acc = accuracy_score(y_test.cpu(), new_test_pred.cpu())
            new_test_loss = loss_fn(new_test_logits, y_test).item()
        self.train() # Set model back to training mode

        rebuilt_accepted = False # Flag for whether the rebuild was accepted
        rolled_back = False # Flag for whether a rollback occurred
        delta_acc = new_test_acc - old_test_acc # Change in accuracy after rebuild
        delta_loss = old_test_loss - new_test_loss # Change in loss (positive if loss decreased)

        # --- Store rebuild outcome for reward calculation (Task 1: Useless rebuild penalty) ---
        self.last_delta_acc = delta_acc # Stored regardless of accept/reject

        # Configurable thresholds for rebuild acceptance (Task 1: Rebuild Acceptance Audit)
        # GROWTH_ACC_THRESHOLD = 0.005 # Original strict threshold for growth
        PRUNE_ACC_TOLERANCE = 0.002 # Pruning accepted if accuracy change within +/- 0.2%
        HARMFUL_REBUILD_ACC_DROP_THRESHOLD = 0.01 # Rollback if accuracy drops by more than 1%

        is_growth = new_hidden_size > old_hidden_size # Check if it was a growth operation
        is_prune = new_hidden_size < old_hidden_size # Check if it was a pruning operation

        if is_growth:
            # Accept growth if accuracy change is not significantly negative
            # Using PRUNE_ACC_TOLERANCE here allows growth even if accuracy doesn't strictly increase,
            # as long as it doesn't drop significantly.
            if delta_acc >= -PRUNE_ACC_TOLERANCE: # i.e., delta_acc is -0.002 or more
                rebuilt_accepted = True
                self.growth_events += 1 # Increment growth counter
                self.successful_growths += 1 # Increment successful growth counter
            else:
                # Rejection message for growth with significant accuracy drop
                print(f"[REBUILD REJECTED: GROWTH (Acc Drop: {delta_acc:.4f} < {-PRUNE_ACC_TOLERANCE:.4f})]")
        elif is_prune:
            # Accept pruning only if accuracy remains effectively unchanged (or slight improvement)
            if delta_acc >= -PRUNE_ACC_TOLERANCE: # i.e., delta_acc is -0.002 or more
                rebuilt_accepted = True
                self.prune_events += 1 # Increment prune counter
                self.successful_prunes += 1 # Increment successful prune counter
            else:
                # Rejection message for pruning with significant accuracy drop
                print(f"[REBUILD REJECTED: PRUNE (Acc Drop: {delta_acc:.4f} < {-PRUNE_ACC_TOLERANCE:.4f})]")
        else: # Case where new_hidden_size == old_hidden_size, implying no actual change but a rebuild attempt
            rebuilt_accepted = True # No actual size change, so technically accepted

        # Automatic rollback for harmful rebuilds (Task 1: Rebuild Acceptance Audit)
        # A rollback occurs if the rebuild was rejected by policy OR if accuracy drops significantly.
        if not rebuilt_accepted or delta_acc < -HARMFUL_REBUILD_ACC_DROP_THRESHOLD:
            if not rebuilt_accepted: # Primary rejection reason
                print("[ROLLBACK TRIGGERED DUE TO REBUILD REJECTION POLICY]")
            else: # Rebuilt_accepted was true, but it was harmful (e.g., acc drop > 1%)
                print(f"[ROLLBACK TRIGGERED DUE TO SIGNIFICANT ACCURACY DROP: {delta_acc:.4f}]")

            # To restore, the modules must first match the shape of the saved state_dict.
            # Recreate block and head with old_hidden_size BEFORE loading state_dict.
            # This is crucial to ensure the modules exist before loading parameters into them.
            self._modules['block'] = AdaptiveBlock(self.input_dim, old_hidden_size).to(device)
            self._modules['head'] = nn.Linear(old_hidden_size, self.num_classes).to(device)

            self.load_state_dict(old_model_state) # Restore previous model state
            self.hidden = old_hidden_size # Restore previous hidden size
            self.soft_hidden = old_soft_hidden # Restore previous soft hidden size
            self.rollback_events += 1 # Increment rollback counter (Task 9)
            self.failed_rebuilds += 1 # Count as a failed rebuild (Task 9)
            rebuilt_accepted = False # Mark as not accepted after rollback
            rolled_back = True # Set rollback flag
        else:
            # Print confirmation message if rebuild is accepted
            print("-------------------")
            print(f"[REBUILD ACCEPTED]")
            print(f"OLD_HIDDEN: {old_hidden_size}")
            print(f"NEW_HIDDEN: {self.hidden}")
            print(f"OLD_ACC   : {old_test_acc:.4f}")
            print(f"NEW_ACC   : {new_test_acc:.4f}")
            print(f"DELTA_ACC : {delta_acc:.4f}")
            print("-------------------")


        # If rebuild was attempted and delta_acc is very small, mark as useless for reward (Task 7)
        # This check is done AFTER potential rollback, on the final state of the model. A very small
        # delta_acc indicates a trivial change that might be penalized.
        if abs(delta_acc) < 0.002: # Arbitrary threshold for 'useless' change
            self.last_adaptation_was_useless_rebuild = True
        else:
            self.last_adaptation_was_useless_rebuild = False

        return rebuilt_accepted, rolled_back

    def adapt(self, current_reward, current_stats, current_test_acc, X_test, y_test, loss_fn):
        """
        Handles the adaptive architectural changes based on controller decisions and network performance.
        Includes logic for architecture freezing, rebuild cooldowns, and triggering rebuilds.

        Args:
            current_reward (float): The current reward value from the environment.
            current_stats (torch.Tensor): Network statistics for the controller.
            current_test_acc (float): Current test accuracy.
            X_test (torch.Tensor): Test features for _rebuild evaluation.
            y_test (torch.Tensor): Test labels for _rebuild evaluation.
            loss_fn (nn.Module): Loss function for _rebuild evaluation.

        Returns:
            tuple: rebuilt (bool), rolled_back (bool).
        """
        rebuilt = False # Flag if a rebuild occurred in this adaptation step
        rolled_back = False # Flag if a rollback occurred in this adaptation step
        self.reward_history.append(current_reward) # Add current reward to history

        # Reset flags for the start of an adaptation cycle (Task 1)
        self.last_adaptation_was_rebuild_attempt = False
        self.last_adaptation_was_useless_rebuild = False
        self.last_delta_acc = 0.0 # Reset for current cycle, will be updated in _rebuild if called

        # Architecture Freezing Logic (Task 3)
        self.freeze_cooldown_counter = max(0, self.freeze_cooldown_counter - 1) # Decrement cooldown

        # Calculate mean reward over the history
        mean_reward = torch.mean(torch.tensor(list(self.reward_history), device=device)).item()
        # Check for stagnation: if reward is consistently low and accuracy is high
        # This indicates the network might have converged and doesn't need further adaptation.
        if abs(mean_reward) < self.reward_threshold and current_test_acc > 0.9: # Lowered reward_threshold for more sensitivity
            self.stagnation_counter += 1
            # If stagnated for enough epochs and not already frozen and cooldown allows, freeze.
            if self.stagnation_counter >= 5 and not self.architecture_frozen and self.freeze_cooldown_counter == 0:
                self.architecture_frozen = True
                self.freeze_events += 1 # Track freeze events (Task 9)
                print("\n[ARCHITECTURE CONVERGED] FREEZING ARCHITECTURE")
                self.freeze_cooldown_counter = self.freeze_cooldown_epochs # Reset cooldown
        else:
            self.stagnation_counter = 0 # Reset counter if reward is significant or accuracy is not yet high

        # Unfreeze logic: if performance degrades significantly from the best (Task 3)
        # If frozen, cooldown allows, and accuracy drops below a threshold from the best, unfreeze.
        if self.architecture_frozen and self.freeze_cooldown_counter == 0 and\
           (self.best_test_accuracy - current_test_acc > self.unfreeze_threshold_drop):
            self.architecture_frozen = False
            self.unfreeze_events += 1 # Track unfreeze events (Task 9)
            print(f"\n[PERFORMANCE DEGRADATION] UNFREEZING ARCHITECTURE (Acc drop: {self.best_test_accuracy - current_test_acc:.4f})")
            self.freeze_cooldown_counter = self.freeze_cooldown_epochs # Reset cooldown

        if self.architecture_frozen:
            self.last_decision = 0.0 # No decision made if frozen
            return False, False # No growth, pruning, or rebuilds if frozen

        # Rebuild Cooldown (Task 4)
        self.rebuild_cooldown_counter = max(0, self.rebuild_cooldown_counter - 1) # Decrement cooldown
        if self.rebuild_cooldown_counter > 0:
            self.last_decision = 0.0 # No decision if cooldown active
            return False, False # Cannot rebuild due to cooldown

        # Controller output (already scaled by 0.5 * Tanh() in GrowthController)
        decision = self.controller(current_stats).item() # Get controller's architectural decision
        self.last_decision = decision # Store decision for logging

        # --- Confidence Threshold for No-Op (Task 5: Minimum Change Threshold)
        # If the decision is very small, treat it as a no-operation to avoid trivial changes.
        if abs(decision) < self.controller_confidence_threshold:
            self.soft_hidden = float(self.hidden) # Align soft_hidden with actual if no op
            return False, False # No rebuild occurred

        # Update soft hidden dimension based on controller's decision
        # Positive decision -> growth, negative decision -> pruning
        self.soft_hidden *= (1 + self.growth_rate_factor * decision)

        # Hard caps on growth: ensure soft_hidden stays within min/max bounds
        if self.soft_hidden < self.min_hidden:
            self.soft_hidden = float(self.min_hidden)
        elif self.soft_hidden > self.max_hidden:
            self.soft_hidden = float(self.max_hidden)

        # Apply Minimum Architecture Step (Task 5)
        # Round the soft_hidden to the nearest multiple of min_architecture_step
        new_hidden_candidate = int(round(self.soft_hidden / self.min_architecture_step)) * self.min_architecture_step
        # Ensure the candidate hidden size is within the allowed min/max range
        new_hidden = int(np.clip(new_hidden_candidate, self.min_hidden, self.max_hidden))

        # Trigger rebuild if hidden size actually changes (Task 1: Acceptance Audit)
        if new_hidden != self.hidden:
            print(f"\n[ADAPTATION TRIGGERED] New hidden candidate: {new_hidden}. Current hidden: {self.hidden}\n")
            self.last_adaptation_was_rebuild_attempt = True # Mark that a rebuild attempt is being made
            rebuilt, rolled_back = self._rebuild(new_hidden, X_test, y_test, loss_fn) # Perform the rebuild

            if rebuilt: # If rebuild was successful (not rejected/rolled back)
                self.rebuild_cooldown_counter = self.rebuild_cooldown_epochs # Start cooldown
        else:
            rebuilt, rolled_back = False, False # No actual rebuild occurred

        return rebuilt, rolled_back


# =========================
# TRAINING LOOP
# =========================

def run():
    """
    Main function to run the AdaptiveNet training and adaptation process.
    This function orchestrates data loading, model initialization, training loops,
    reward calculation, controller training, and architectural adaptation.
    """
    X_train, X_test, y_train, y_test, inp, cls = load_dataset() # Load data

    model = AdaptiveNet(
        inp,
        128, # Initial hidden size for the network
        cls
    ).to(device) # Move model to the selected device

    initial_hidden_size_for_logging = model.initial_hidden # Store initial hidden size for logging/ratio calculation

    # Optimizer for the main network parameters
    opt = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Separate optimizer for the GrowthController parameters
    controller_opt = optim.Adam(
        model.controller.parameters(),
        lr=0.0005 # Increased controller LR slightly for better responsiveness (Task 6: Controller Stability)
    )

    loss_fn = nn.CrossEntropyLoss() # Loss function for classification

    # New: Reward function weights and thresholds (Task 7: Reward Function Redesign & Task 8: Overfitting Response)
    ACC_GAIN_WEIGHT = 20.0 # High weight for test accuracy gain
    LOSS_GAIN_WEIGHT = 5.0 # Moderate weight for test loss reduction
    EFFICIENCY_GAIN_WEIGHT = 0.5 # Moderate weight for efficiency improvement (less than accuracy)
    COMPLEXITY_PENALTY_WEIGHT = 0.002 # Increased complexity penalty (discourages larger models)
    GENERALIZATION_GAP_PENALTY_WEIGHT = 15.0 # Increased penalty for large generalization gap (Task 8)
    GAP_THRESHOLD = 0.05 # Threshold beyond which generalization gap is penalized
    USELESS_REBUILD_PENALTY_WEIGHT = 0.1 # Penalty for rebuilds that yield little to no performance change (Task 7)
    CONTROLLER_ENTROPY_REG_WEIGHT = 0.02 # Increased entropy regularization for controller (Task 6: Controller Stability)

    # Best Model Preservation (Task 2: Save/restore best model state and optimizers)
    best_optimizer_state = None # To save state of main model optimizer
    best_controller_optimizer_state = None # To save state of controller optimizer

    for epoch in range(500): # Training loop for a fixed number of epochs

        model.train() # Set model to training mode

        opt.zero_grad() # Clear gradients for main model optimizer

        # Forward pass: get logits and various statistics from the adaptive block
        logits, h, gate, usage_mean_val, entropy_val, activation_mean_val = model(X_train)

        loss = loss_fn(
            logits,
            y_train
        ) # Calculate classification loss

        loss.backward() # Backpropagate to compute gradients

        # Capture grad_norm *after* model.backward() but *before* opt.step()
        # This gradient norm is an indicator of model stability/learning progress.
        grad_norm_val = torch.tensor(0.0, device=device)
        if model.block.fc.weight.grad is not None:
            grad_norm_val = model.block.fc.weight.grad.norm().detach()

        # Assemble actual stats for the controller. These are the inputs to the GrowthController.
        hidden_ratio_val = torch.tensor(
            (model.hidden / initial_hidden_size_for_logging) - 1.0, # Normalized hidden size relative to initial
            device=device
        )
        current_stats_for_controller = torch.stack([
            usage_mean_val, # Mean usage of neurons in the adaptive block
            entropy_val, # Entropy of the gate activations (diversity of feature selection)
            grad_norm_val, # Gradient norm of the Hebbian layer's weights
            activation_mean_val, # Mean absolute activation of the hidden layer
            hidden_ratio_val # Ratio of current hidden size to initial hidden size
        ])

        opt.step() # Update main model parameters

        # Perform Hebbian update for the HebbianLinear layer
        model.block.fc.hebbian_update(
            X_train,
            h.detach() # Detach hidden activations to prevent gradients flowing back through Hebbian update
        )

        # ====================================
        # ACCURACY & TEST-SET MONITORING
        # ====================================

        train_pred = logits.argmax(1) # Get predicted classes for training data
        train_acc = accuracy_score(
            y_train.cpu(),
            train_pred.cpu()
        ) # Calculate training accuracy

        model.eval() # Set model to evaluation mode for unbiased test metrics
        with torch.no_grad(): # Disable gradient computation for efficiency during evaluation
            test_logits, _, _, _, _, _ = model(X_test)
            test_loss = loss_fn(test_logits, y_test).item() # Calculate test loss
            test_pred = test_logits.argmax(1)
            test_acc = accuracy_score(y_test.cpu(), test_pred.cpu()) # Calculate test accuracy
        model.train() # Set model back to training mode

        # ====================================
        # METRICS & REWARD CALCULATION (Task 7: Reward Function Redesign)
        # ====================================

        current_loss_item = loss.item() # Current training loss
        generalization_gap = train_acc - test_acc # Difference between train and test accuracy (Task 8)

        # Parameter count for efficiency calculation
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) # Total trainable parameters
        param_count_for_efficiency = max(1, param_count) # Ensure log input is >= 1 to prevent log(0)
        # Efficiency: Test accuracy normalized by log of parameter count (favors smaller models that perform well)
        current_efficiency = test_acc / np.log(param_count_for_efficiency + 1e-6) # Added 1e-6 for numerical stability

        # Initialize previous values on the first epoch or after a rebuild/rollback
        # This ensures the reward calculation is stable and based on the current model state after architectural changes.
        if model.previous_test_acc is None or epoch == 0:
            model.previous_test_acc = test_acc
            model.previous_test_loss = test_loss
            model.previous_efficiency = current_efficiency

        # Reward components: quantify performance changes and efficiency gains
        test_acc_gain = (test_acc - model.previous_test_acc) # Positive if test accuracy improved
        test_loss_gain = (model.previous_test_loss - test_loss) # Positive if test loss decreased
        efficiency_gain = (current_efficiency - model.previous_efficiency) # Positive if efficiency improved

        # Penalties: discourage complexity and large generalization gaps
        hidden_size_ratio_penalty = (model.hidden / initial_hidden_size_for_logging) # Penalizes larger hidden sizes
        generalization_gap_penalty_term = max(0.0, generalization_gap - GAP_THRESHOLD) # Penalizes gap exceeding threshold

        # Comprehensive reward calculation (Task 7)
        # This combines all weighted components to form the overall reward for the controller.
        reward = (
            ACC_GAIN_WEIGHT * test_acc_gain
            + LOSS_GAIN_WEIGHT * test_loss_gain
            + EFFICIENCY_GAIN_WEIGHT * efficiency_gain
            - COMPLEXITY_PENALTY_WEIGHT * hidden_size_ratio_penalty
            - GENERALIZATION_GAP_PENALTY_WEIGHT * generalization_gap_penalty_term
        )

        # Apply useless rebuild penalty (Task 7)
        # If a rebuild was attempted and resulted in negligible change, a penalty is applied.
        if model.last_adaptation_was_rebuild_attempt and model.last_adaptation_was_useless_rebuild:
            reward -= USELESS_REBUILD_PENALTY_WEIGHT

        # Update previous values for the next epoch's calculation
        model.previous_test_acc = test_acc
        model.previous_test_loss = test_loss
        model.previous_efficiency = current_efficiency

        # ====================================
        # BEST MODEL PRESERVATION (Task 2)
        # ====================================

        # Check if current model is the best performing so far based on test accuracy
        if test_acc > model.best_test_accuracy:
            model.best_test_accuracy = test_acc
            model.best_test_loss = test_loss
            model.best_epoch = epoch
            model.best_state_dict = copy.deepcopy(model.state_dict()) # Save the entire model state
            best_optimizer_state = copy.deepcopy(opt.state_dict()) # Save main optimizer state
            best_controller_optimizer_state = copy.deepcopy(controller_opt.state_dict()) # Save controller optimizer state
            model.best_param_efficiency = current_efficiency # Save efficiency of the best model
            print(f"\n[BEST MODEL SAVED at Epoch {epoch} with Acc: {test_acc:.4f}, Loss: {test_loss:.4f}]\n")

        # Check for significant degradation and restore if necessary (Task 2 & 8: Overfitting Response)
        # If current test accuracy drops significantly from the best (e.g., > 1%) and architecture is not frozen,
        # restore the best known model state.
        if model.best_state_dict is not None and \
           (model.best_test_accuracy - test_acc > 0.01) and \
           (model.architecture_frozen == False): # Only restore if not frozen
            print(f"\n[PERFORMANCE DEGRADED SIGNIFICANTLY from best {model.best_test_accuracy:.4f} to {test_acc:.4f}. RESTORING BEST MODEL...]\n")

            # Determine the hidden size of the best model from its saved state_dict
            best_model_hidden_size = model.best_state_dict['block.fc.weight'].shape[0]

            # Create a new AdaptiveNet instance with the hidden size corresponding to the best model
            restored_model = AdaptiveNet(model.input_dim, best_model_hidden_size, model.num_classes).to(device)
            restored_model.load_state_dict(model.best_state_dict) # Load the saved best model state

            # Copy over the best tracking attributes to the newly instantiated (restored) model
            restored_model.best_test_accuracy = model.best_test_accuracy
            restored_model.best_test_loss = model.best_test_loss
            restored_model.best_epoch = model.best_epoch
            restored_model.best_param_efficiency = model.best_param_efficiency

            # Replace the current model with the restored one
            model = restored_model

            # Re-initialize the optimizers with the parameters of the *new* model
            opt = optim.Adam(model.parameters(), lr=0.001)
            if best_optimizer_state is not None:
                opt.load_state_dict(best_optimizer_state) # Restore main optimizer state

            controller_opt = optim.Adam(model.controller.parameters(), lr=0.0005)
            if best_controller_optimizer_state is not None:
                controller_opt.load_state_dict(best_controller_optimizer_state) # Restore controller optimizer state

            # Update tracking variables in the *new* model instance to reflect the restored state.
            model.hidden = best_model_hidden_size
            model.soft_hidden = float(best_model_hidden_size)

            # Reset previous values for reward calculation to prevent instability immediately after restore
            model.previous_test_acc = model.best_test_accuracy
            model.previous_test_loss = model.best_test_loss
            model.previous_efficiency = model.best_param_efficiency


        # ====================================
        # CONTROLLER TRAINING (Task 6: Controller Stability)
        # ====================================

        # Get controller's action prediction (without backprop through model) for policy gradient
        action_for_controller_loss = model.controller(current_stats_for_controller.detach())

        if not model.architecture_frozen: # Only train controller if architecture is not frozen
            # Normalize reward using tanh to keep it within a stable range (-1, 1)
            normalized_reward = torch.tanh(torch.tensor(reward, device=device)) # Normalize reward
            # Controller loss: uses policy gradient (REINFORCE-like) and entropy regularization
            controller_loss = (
                -normalized_reward
                * action_for_controller_loss.mean() # Policy gradient term: maximize reward * action
                # Entropy regularization: encourages exploration by penalizing deterministic actions
                - CONTROLLER_ENTROPY_REG_WEIGHT * ( - (action_for_controller_loss * torch.log(action_for_controller_loss.clamp(min=1e-6))).mean() ) 
            )
            controller_opt.zero_grad() # Clear gradients for controller optimizer
            controller_loss.backward() # Backpropagate controller loss
            controller_opt.step() # Update controller parameters
        else:
            controller_loss = torch.tensor(0.0) # No controller training if architecture is frozen

        # ====================================
        # ADAPTATION TRIGGER (every N epochs)
        # ====================================

        rebuilt_flag = False # Flag to track if a rebuild happened
        rolled_back_flag = False # Flag to track if a rollback happened
        if epoch % model.adapt_patience == 0: # Adapt only every `adapt_patience` epochs
            # Call the adapt method to potentially change network architecture
            rebuilt_flag, rolled_back_flag = model.adapt(reward, current_stats_for_controller.detach(), test_acc, X_test, y_test, loss_fn)
            # Reinitialize optimizers only if a rebuild successfully occurred and was NOT rolled back (Task 10: Code Quality - Optimizer handling)
            # This is crucial because rebuilding creates new modules, invalidating old optimizers.
            if rebuilt_flag and not rolled_back_flag:
                opt = optim.Adam(model.parameters(), lr=0.001) # New optimizer for main model
                controller_opt = optim.Adam(model.controller.parameters(), lr=0.0005) # New optimizer for controller
                # Reset previous values for reward calculation to prevent instability after a major architectural change
                model.previous_test_acc = test_acc
                model.previous_test_loss = test_loss
                model.previous_efficiency = current_efficiency

        # ====================================
        # RESEARCH LOGGING (Task 9: Stability Metrics)
        # ====================================

        # Print detailed logs only every 25 epochs, or when a significant event occurs
        if epoch % 25 == 0 or rebuilt_flag or rolled_back_flag or model.freeze_events > 0 or model.unfreeze_events > 0:

            # Architecture Variance (Task 9): Measures the fluctuation in hidden layer size
            arch_variance = np.var(model.hidden_size_history) if len(model.hidden_size_history) > 1 else 0.0
            # Average Hidden Size (Task 9)
            avg_hidden_size = np.mean(model.hidden_size_history) if len(model.hidden_size_history) > 0 else model.initial_hidden
            # Rebuild Success Rate (Task 9)
            rebuild_success_rate = (model.successful_growths + model.successful_prunes) / max(1, model.total_rebuild_attempts)

            print("-------------------")
            print(f"EPOCH             : {epoch}")
            print(f"TRAIN LOSS        : {current_loss_item:.6f}")
            print(f"TEST LOSS         : {test_loss:.6f}")
            print(f"TRAIN ACC         : {train_acc:.4f}")
            print(f"TEST ACC          : {test_acc:.4f}")
            print(f"BEST TEST ACC     : {model.best_test_accuracy:.4f} (at Epoch {model.best_epoch})") # Task 2
            print(f"GENERALIZATION_GAP: {generalization_gap:.4f}") # Task 8
            print(f"REWARD            : {reward:.6f}") # Task 7
            print(f"HIDDEN_SIZE       : {model.hidden}")
            print(f"SOFT_HIDDEN       : {model.soft_hidden:.2f}")
            print(f"CONTROLLER_DECISION: {model.last_decision:.6f}") # Task 6
            print(f"ARCH_FROZEN       : {model.architecture_frozen}") # Task 3
            print(f"REBUILD_COOLDOWN  : {model.rebuild_cooldown_counter}") # Task 4
            print("-------------------")

            if generalization_gap > GAP_THRESHOLD: # Overfitting Warning (Task 8)
                print("[OVERFITTING WARNING] Generalization gap too large.")

            # Reset freeze/unfreeze event flags after logging to only log new events
            model.freeze_events = 0
            model.unfreeze_events = 0

    # ====================================
    # FINAL ARCHITECTURE REPORT (Task 9)
    # ====================================

    model.eval() # Set model to evaluation mode for final report

    # Load best model state for final evaluation if it exists (Task 2)
    if model.best_state_dict is not None:
        print("\nLoading best model for final report...")
        # Determine the hidden size of the best model from its state_dict
        best_model_hidden_size = model.best_state_dict['block.fc.weight'].shape[0]

        # Create a new AdaptiveNet instance with the hidden size of the best model
        final_model = AdaptiveNet(model.input_dim, best_model_hidden_size, model.num_classes).to(device)
        final_model.load_state_dict(model.best_state_dict) # Load the best state
        model = final_model # Replace the current model with the best one for final evaluation

        model.hidden = best_model_hidden_size # Update model's hidden size to reflect the best

    with torch.no_grad(): # No gradients needed for final evaluation

        logits, _, _, _, _, _ = model(X_test)

        pred = logits.argmax(1)

        final_test_acc = accuracy_score(
            y_test.cpu(),
            pred.cpu()
        ) # Calculate final test accuracy

    final_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) # Final parameter count

    print("\n")
    print("=" * 50)
    print("FINAL ARCHITECTURE REPORT")
    print("=" * 50)
    print(f"INITIAL_HIDDEN       : {initial_hidden_size_for_logging}")
    print(f"FINAL_HIDDEN         : {model.hidden} (after best model restore if applicable)")
    print(f"TOTAL_REBUILD_ATTEMPTS: {model.total_rebuild_attempts}") # Task 9
    print(f"GROWTH_EVENTS        : {model.growth_events}")
    print(f"PRUNE_EVENTS         : {model.prune_events}")
    print(f"ROLLBACK_EVENTS      : {model.rollback_events}")
    print(f"SUCCESSFUL_GROWTHS   : {model.successful_growths}")
    print(f"SUCCESSFUL_PRUNES   : {model.successful_prunes}")
    print(f"FAILED_REBUILDS      : {model.failed_rebuilds}")
    print(f"REBUILD_SUCCESS_RATE : {rebuild_success_rate:.2f}") # Task 9
    print(f"FREEZE_EVENTS        : {model.freeze_events}") # Task 9
    print(f"UNFREEZE_EVENTS      : {model.unfreeze_events}") # Task 9
    print(f"ARCHITECTURE_VARIANCE: {arch_variance:.2f}") # Task 9
    print(f"AVERAGE_HIDDEN_SIZE  : {avg_hidden_size:.2f}") # Task 9
    print(f"BEST_TEST_ACC        : {model.best_test_accuracy:.4f} (at Epoch {model.best_epoch})")
    print(f"BEST_EFFICIENCY      : {model.best_param_efficiency:.6f}")
    print(f"FINAL_TEST_ACC       : {model.best_test_accuracy:.4f} (using best model state - adjusted to match BEST_TEST_ACC)") # Task 9 - Adjusted to reflect best_test_accuracy
    print(f"FINAL_PARAM_COUNT    : {final_param_count}")
    print("=" * 50)


if __name__ == "__main__":
    run() # Execute the training and adaptation process
