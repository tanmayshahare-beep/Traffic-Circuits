import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# 1. ENHANCED DATA GENERATION
# ============================================================================

class TrafficCircuitDataGenerator:
    """Generate synthetic traffic circuit data with realistic patterns"""
    
    def __init__(self, n_samples=5000, n_nodes=10, n_links=15):
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.n_links = n_links
        
        # Define traffic patterns
        self.patterns = {
            'morning_peak': {'peak_hour': 8, 'intensity': 1.8, 'duration': 3},
            'midday': {'peak_hour': 13, 'intensity': 1.2, 'duration': 4},
            'evening_peak': {'peak_hour': 18, 'intensity': 2.0, 'duration': 3},
            'night': {'peak_hour': 22, 'intensity': 0.7, 'duration': 6},
            'rainy': {'intensity': 2.5, 'variability': 0.3},
            'accident': {'probability': 0.05, 'severity': 3.0},
            'construction': {'probability': 0.03, 'severity': 2.0}
        }
        
    def generate_dataset(self):
        """Generate comprehensive traffic dataset"""
        
        # Time features
        hours = np.random.uniform(0, 24, self.n_samples)
        days = np.random.randint(1, 8, self.n_samples)  # 1-7 for days of week
        months = np.random.randint(1, 13, self.n_samples)
        
        # Weather conditions
        weather = np.random.choice(['clear', 'rain', 'snow', 'fog'], 
                                  self.n_samples, p=[0.6, 0.25, 0.1, 0.05])
        temperature = np.random.normal(20, 10, self.n_samples)
        visibility = np.random.uniform(0.5, 10, self.n_samples)
        
        # Traffic flow rates (vehicles/hour)
        base_flows = np.random.lognormal(5, 0.3, (self.n_samples, self.n_links))
        
        # Apply time patterns
        for i, hour in enumerate(hours):
            # Morning peak (7-10 AM)
            if 7 <= hour <= 10:
                base_flows[i] *= 1.5 + 0.5 * np.sin((hour - 7) * np.pi / 3)
            # Evening peak (5-8 PM)
            elif 17 <= hour <= 20:
                base_flows[i] *= 1.7 + 0.3 * np.sin((hour - 17) * np.pi / 3)
            # Night (10 PM - 5 AM)
            elif 22 <= hour or hour <= 5:
                base_flows[i] *= 0.4 + 0.2 * np.random.rand()
                
            # Weekend effect
            if days[i] >= 6:  # Saturday or Sunday
                if 10 <= hour <= 18:
                    base_flows[i] *= 1.3  # More traffic during weekend days
                else:
                    base_flows[i] *= 0.8  # Less traffic at night
        
        # Weather effects
        for i, w in enumerate(weather):
            if w == 'rain':
                base_flows[i] *= 0.8  # Reduced flow
            elif w == 'snow':
                base_flows[i] *= 0.6
            elif w == 'fog':
                base_flows[i] *= 0.7
        
        # Generate link characteristics
        link_lengths = np.random.uniform(0.5, 10, self.n_links)  # km
        link_lanes = np.random.choice([1, 2, 3, 4], self.n_links, p=[0.1, 0.4, 0.4, 0.1])
        link_speed_limits = np.random.choice([40, 60, 80, 100], self.n_links)
        
        # Generate node connectivity (random graph)
        node_connections = []
        link_node_pairs = []
        
        # Create a connected graph
        for i in range(self.n_nodes - 1):
            node_connections.append((i, i + 1))
        # Add some cross connections
        extra_connections = np.random.randint(0, self.n_nodes, (self.n_links - (self.n_nodes - 1), 2))
        extra_connections = [(min(a,b), max(a,b)) for a,b in extra_connections]
        extra_connections = list(set(extra_connections))
        
        node_connections.extend(extra_connections[:self.n_links - (self.n_nodes - 1)])
        link_node_pairs = node_connections[:self.n_links]
        
        # Calculate travel times using BPR function
        capacities = link_lanes * 1000  # vehicles/hour per lane
        free_flow_times = link_lengths / (link_speed_limits / 3.6) * 60  # minutes
        
        # Bureau of Public Roads (BPR) function
        alphas = np.random.uniform(0.15, 0.25, self.n_links)
        betas = np.random.uniform(4, 6, self.n_links)
        
        travel_times = np.zeros((self.n_samples, self.n_links))
        for i in range(self.n_links):
            flow_ratio = base_flows[:, i] / capacities[i]
            travel_times[:, i] = free_flow_times[i] * (1 + alphas[i] * (flow_ratio ** betas[i]))
        
        # Add random incidents
        incidents = np.random.binomial(1, 0.05, (self.n_samples, self.n_links))
        incident_severity = np.random.uniform(1.5, 4.0, (self.n_samples, self.n_links))
        travel_times *= (1 + incidents * incident_severity)
        
        # Create DataFrame
        data = {}
        
        # Time features
        data['hour'] = hours
        data['day_of_week'] = days
        data['month'] = months
        
        # Weather features
        data['weather'] = weather
        data['temperature'] = temperature
        data['visibility'] = visibility
        
        # Link features
        for i in range(self.n_links):
            data[f'flow_link_{i}'] = base_flows[:, i]
            data[f'travel_time_link_{i}'] = travel_times[:, i]
            data[f'congestion_link_{i}'] = base_flows[:, i] / capacities[i]
            
        # Node travel times (sum of incident links)
        for node in range(self.n_nodes):
            connected_links = [j for j, (a,b) in enumerate(link_node_pairs) if node in [a,b]]
            if connected_links:
                data[f'node_{node}_time'] = np.mean(travel_times[:, connected_links], axis=1)
            else:
                data[f'node_{node}_time'] = np.zeros(self.n_samples)
        
        # Target: total network delay
        total_delay = np.sum(travel_times - free_flow_times, axis=1)
        data['total_delay'] = total_delay
        data['avg_speed'] = np.sum(link_lengths) / (np.sum(travel_times, axis=1) / 60)  # km/h
        
        return pd.DataFrame(data), link_node_pairs, capacities, free_flow_times

# ============================================================================
# 2. PINN MODEL FOR TRAFFIC CIRCUIT
# ============================================================================

class TrafficPINN(nn.Module):
    """Physics-Informed Neural Network for traffic flow prediction"""
    
    def __init__(self, input_dim, hidden_dim=128, n_layers=5):
        super(TrafficPINN, self).__init__()
        
        # Main neural network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Physics-informed components
        self.alpha = nn.Parameter(torch.tensor(0.15))  # BPR alpha
        self.beta = nn.Parameter(torch.tensor(4.0))    # BPR beta
        
    def forward(self, x):
        return self.network(x)
    
    def physics_loss(self, flow, capacity, free_flow_time, total_delay_pred):
        """Physics loss based on BPR function"""
        
        # Ensure positive values
        flow = torch.clamp(flow, min=1e-6)
        capacity = torch.clamp(capacity, min=1e-6)
        
        # Calculate expected travel time from BPR for each link
        flow_ratio = flow / capacity
        link_travel_time = free_flow_time * (1 + torch.abs(self.alpha) * (flow_ratio ** torch.abs(self.beta)))
        
        # Calculate expected total delay from physics
        link_delay = link_travel_time - free_flow_time
        physics_total_delay = torch.sum(link_delay, dim=1, keepdim=True)
        
        # Physics mismatch loss
        physics_loss = torch.mean((total_delay_pred - physics_total_delay) ** 2)
        
        return physics_loss
    
    def conservation_loss(self, node_flows_in, node_flows_out):
        """Flow conservation at nodes"""
        conservation_loss = torch.mean((node_flows_in - node_flows_out) ** 2)
        return conservation_loss

# ============================================================================
# 3. TRAFFIC CIRCUIT SIMULATOR
# ============================================================================

class TrafficCircuitSimulator:
    """Simulate traffic using electrical circuit analogy"""
    
    def __init__(self, n_nodes, link_pairs, resistances=None):
        self.n_nodes = n_nodes
        self.link_pairs = link_pairs
        self.n_links = len(link_pairs)
        
        if resistances is None:
            self.resistances = np.random.uniform(0.5, 2.0, self.n_links)
        else:
            self.resistances = resistances
            
        # Create incidence matrix
        self.incidence_matrix = self._create_incidence_matrix()
        
    def _create_incidence_matrix(self):
        """Create node-link incidence matrix"""
        A = np.zeros((self.n_nodes, self.n_links))
        for link_idx, (i, j) in enumerate(self.link_pairs):
            A[i, link_idx] = 1
            A[j, link_idx] = -1
        return A
    
    def solve_circuit(self, current_sources):
        """
        Solve circuit using nodal analysis
        current_sources: current injected at each node
        """
        # Remove ground node (last node)
        A_reduced = self.incidence_matrix[:-1, :]
        
        # Conductance matrix
        G = np.diag(1.0 / self.resistances)
        
        # Admittance matrix
        Y = A_reduced @ G @ A_reduced.T
        
        # Solve for node voltages
        I = current_sources[:-1]  # Remove ground
        try:
            V = np.linalg.solve(Y, I)
        except np.linalg.LinAlgError:
            V = np.linalg.lstsq(Y, I, rcond=None)[0]
        
        # Add ground voltage (0)
        V_full = np.zeros(self.n_nodes)
        V_full[:-1] = V
        
        # Calculate currents through links
        voltages_diff = np.array([V_full[j] - V_full[i] for i, j in self.link_pairs])
        link_currents = voltages_diff / self.resistances
        
        # Calculate travel times (proportional to voltage drops)
        travel_times = np.abs(voltages_diff) * 10  # Scaling factor
        
        return V_full, link_currents, travel_times
    
    def simulate_traffic(self, demands, time_of_day='midday', weather='clear'):
        """
        Simulate traffic with time and weather effects
        """
        # Adjust resistances based on conditions
        resistance_multipliers = np.ones(self.n_links)
        
        if time_of_day == 'morning_peak':
            resistance_multipliers *= 1.8
        elif time_of_day == 'evening_peak':
            resistance_multipliers *= 2.0
        elif time_of_day == 'night':
            resistance_multipliers *= 0.7
            
        if weather == 'rain':
            resistance_multipliers *= 1.5
        elif weather == 'snow':
            resistance_multipliers *= 2.0
        elif weather == 'fog':
            resistance_multipliers *= 1.3
            
        adjusted_resistances = self.resistances * resistance_multipliers
        
        # Create temporary simulator with adjusted resistances
        temp_simulator = TrafficCircuitSimulator(self.n_nodes, self.link_pairs, adjusted_resistances)
        
        # Solve circuit
        V, currents, travel_times = temp_simulator.solve_circuit(demands)
        
        return {
            'voltages': V,
            'currents': currents,
            'travel_times': travel_times,
            'total_delay': np.sum(travel_times),
            'congestion_index': np.mean(currents / np.max(currents)) if np.max(currents) > 0 else 0
        }

# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_traffic_circuit(voltages, currents, link_pairs, title="Traffic Circuit Simulation"):
    """Visualize the traffic circuit"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Circuit diagram (simplified)
    ax1 = axes[0, 0]
    n_nodes = len(voltages)
    
    # Create circular layout
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Plot nodes
    ax1.scatter(x, y, s=300, c=voltages, cmap='viridis', edgecolors='black', linewidth=2)
    
    # Plot links with thickness proportional to current
    max_current = np.max(np.abs(currents))
    for (i, j), current in zip(link_pairs, currents):
        if max_current > 0:
            linewidth = 1 + 3 * abs(current) / max_current
        else:
            linewidth = 1
        ax1.plot([x[i], x[j]], [y[i], y[j]], 'gray', linewidth=linewidth, alpha=0.7)
    
    ax1.set_title(f'Traffic Circuit\nNode Voltages = Travel Time Impedance')
    ax1.axis('equal')
    ax1.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(voltages)
    plt.colorbar(sm, ax=ax1, label='Voltage (Travel Time Impedance)')
    
    # 2. Voltage distribution
    ax2 = axes[0, 1]
    nodes = np.arange(n_nodes)
    ax2.bar(nodes, voltages, color='skyblue', edgecolor='navy')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title('Node Voltages Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Current distribution
    ax3 = axes[1, 0]
    link_indices = np.arange(len(currents))
    colors = ['red' if c > 0 else 'blue' for c in currents]
    ax3.bar(link_indices, currents, color=colors, edgecolor='black')
    ax3.set_xlabel('Link Index')
    ax3.set_ylabel('Current (Traffic Flow)')
    ax3.set_title('Link Currents (Positive = Direction)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Travel times
    ax4 = axes[1, 1]
    travel_times = np.abs(np.array([voltages[j] - voltages[i] for i, j in link_pairs])) * 10
    ax4.bar(link_indices, travel_times, color='orange', edgecolor='darkorange')
    ax4.set_xlabel('Link Index')
    ax4.set_ylabel('Travel Time (min)')
    ax4.set_title('Predicted Travel Times per Link')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_training_history(history):
    """Plot training and validation losses"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Physics loss
    axes[1].plot(history['physics_loss'], label='Physics Loss', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Physics Loss')
    axes[1].set_title('Physics Constraint Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # BPR parameters
    axes[2].plot(history['alpha'], label='α (BPR parameter)', color='green', linewidth=2)
    axes[2].plot(history['beta'], label='β (BPR parameter)', color='purple', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Parameter Value')
    axes[2].set_title('Learned Physics Parameters')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 5. SIMPLIFIED PINN TRAINING
# ============================================================================

def train_pinn_simplified(pinn, X_train, y_train, X_val, y_val, flow_features, 
                         capacities, free_flow_times, n_epochs=300):
    """Simplified training loop for PINN"""
    
    optimizer = optim.AdamW(pinn.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'physics_loss': [], 'alpha': [], 'beta': []
    }
    
    for epoch in range(n_epochs):
        # Training
        pinn.train()
        
        # Random batch
        idx = np.random.choice(len(X_train), min(256, len(X_train)), replace=False)
        batch_X = torch.FloatTensor(X_train[idx])
        batch_y = torch.FloatTensor(y_train[idx])
        
        optimizer.zero_grad()
        
        # Data loss
        predictions = pinn(batch_X)
        data_loss = nn.MSELoss()(predictions, batch_y)
        
        # Physics loss - simplified approach
        batch_flows = batch_X[:, :len(flow_features)]
        batch_capacities = torch.FloatTensor(capacities).unsqueeze(0).expand(batch_flows.shape[0], -1)
        batch_free_flow_times = torch.FloatTensor(free_flow_times).unsqueeze(0).expand(batch_flows.shape[0], -1)
        
        physics_loss = pinn.physics_loss(
            batch_flows, 
            batch_capacities, 
            batch_free_flow_times,
            predictions  # Use predictions directly
        )
        
        # Total loss
        total_loss = data_loss + 0.05 * physics_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        pinn.eval()
        with torch.no_grad():
            val_predictions = pinn(torch.FloatTensor(X_val))
            val_loss = nn.MSELoss()(val_predictions, torch.FloatTensor(y_val)).item()
            
            # Validation physics loss
            val_flows = torch.FloatTensor(X_val[:, :len(flow_features)])
            val_physics_loss = pinn.physics_loss(
                val_flows,
                torch.FloatTensor(capacities).unsqueeze(0).expand(val_flows.shape[0], -1),
                torch.FloatTensor(free_flow_times).unsqueeze(0).expand(val_flows.shape[0], -1),
                val_predictions
            ).item()
        
        # Record history
        history['train_loss'].append(data_loss.item())
        history['val_loss'].append(val_loss)
        history['physics_loss'].append(val_physics_loss)
        history['alpha'].append(pinn.alpha.item())
        history['beta'].append(pinn.beta.item())
        
        scheduler.step(val_loss)
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch:4d} | Train Loss: {data_loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | α: {pinn.alpha.item():.3f} | β: {pinn.beta.item():.3f}")
    
    return history

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    print("🚦 Traffic Circuit Model with PINN - Enhanced Version")
    print("=" * 60)
    
    # Generate enhanced dataset
    print("\n1. Generating enhanced traffic dataset...")
    generator = TrafficCircuitDataGenerator(n_samples=5000, n_nodes=8, n_links=12)
    df, link_pairs, capacities, free_flow_times = generator.generate_dataset()
    
    print(f"   Generated {len(df)} samples")
    print(f"   Network: {len(set([n for pair in link_pairs for n in pair]))} nodes, {len(link_pairs)} links")
    print(f"   Features: {len(df.columns)} columns")
    
    # Display sample data
    print("\n   Sample data (first 5 rows):")
    print(df[['hour', 'weather', 'flow_link_0', 'travel_time_link_0', 'total_delay']].head())
    print(f"\n   Total delay stats - Min: {df['total_delay'].min():.2f}, "
          f"Max: {df['total_delay'].max():.2f}, Mean: {df['total_delay'].mean():.2f}")
    
    # Prepare data for PINN
    print("\n2. Preparing data for Physics-Informed Neural Network...")
    
    # Select features - use all link flows as input
    flow_features = [f for f in df.columns if 'flow_link' in f]
    print(f"   Using {len(flow_features)} flow features")
    
    X = df[flow_features].values
    y = df['total_delay'].values.reshape(-1, 1)
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
    
    print(f"   Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Initialize PINN
    print("\n3. Initializing Physics-Informed Neural Network...")
    input_dim = X_train.shape[1]
    pinn = TrafficPINN(input_dim=input_dim, hidden_dim=128, n_layers=4)
    print(f"   Model architecture: Input={input_dim}, Hidden=128, Layers=4")
    print(f"   Trainable parameters: {sum(p.numel() for p in pinn.parameters() if p.requires_grad):,}")
    
    # Train PINN with simplified approach
    print("\n4. Training PINN with physics constraints...")
    history = train_pinn_simplified(
        pinn, X_train, y_train, X_val, y_val, 
        flow_features, capacities, free_flow_times,
        n_epochs=200
    )
    
    print("\n5. Running traffic circuit simulations...")
    
    # Create circuit simulator
    n_nodes = 8
    simulator = TrafficCircuitSimulator(n_nodes, link_pairs[:n_nodes*2])
    
    # Simulate different scenarios
    scenarios = [
        ('Morning Peak', 'morning_peak', 'clear'),
        ('Midday Normal', 'midday', 'clear'),
        ('Evening Peak', 'evening_peak', 'clear'),
        ('Rainy Morning', 'morning_peak', 'rain'),
        ('Night', 'night', 'clear')
    ]
    
    results = []
    
    for scenario_name, time_of_day, weather in scenarios:
        # Generate random demands
        demands = np.random.randn(n_nodes)
        demands[-1] = -np.sum(demands[:-1])  # Ground node
        
        # Run simulation
        result = simulator.simulate_traffic(demands, time_of_day, weather)
        result['scenario'] = scenario_name
        results.append(result)
        
        print(f"   {scenario_name:20} | Total Delay: {result['total_delay']:6.1f} min | "
              f"Congestion: {result['congestion_index']:.3f}")
    
    # Visualization
    print("\n6. Generating visualizations...")
    
    # Plot training history
    fig1 = plot_training_history(history)
    fig1.savefig('pinn_training_history.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: pinn_training_history.png")
    
    # Plot circuit for each scenario
    for i, (scenario_name, time_of_day, weather) in enumerate(scenarios):
        demands = np.random.randn(n_nodes)
        demands[-1] = -np.sum(demands[:-1])
        
        result = simulator.simulate_traffic(demands, time_of_day, weather)
        fig = plot_traffic_circuit(
            result['voltages'], 
            result['currents'], 
            link_pairs[:n_nodes*2],
            title=f"Traffic Circuit: {scenario_name}"
        )
        filename = f'traffic_circuit_{scenario_name.replace(" ", "_").lower()}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ Saved: {filename}")
    
    # Plot comprehensive analysis
    print("\n7. Generating comprehensive analysis plots...")
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Traffic flow distribution
    axes[0, 0].hist(df[[f for f in df.columns if 'flow_link' in f]].values.flatten(), 
                   bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Traffic Flow (veh/hr)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Traffic Flow Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Travel time distribution
    axes[0, 1].hist(df[[f for f in df.columns if 'travel_time_link' in f]].values.flatten(), 
                   bins=50, color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Travel Time (min)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Travel Time Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Delay by hour of day
    df['hour_int'] = df['hour'].astype(int)
    delay_by_hour = df.groupby('hour_int')['total_delay'].mean()
    axes[0, 2].plot(delay_by_hour.index, delay_by_hour.values, 
                   marker='o', linewidth=2, color='green', markersize=4)
    axes[0, 2].set_xlabel('Hour of Day')
    axes[0, 2].set_ylabel('Average Delay (min)')
    axes[0, 2].set_title('Delay Pattern by Hour of Day')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xticks(range(0, 24, 3))
    
    # 4. Weather impact
    weather_delay = df.groupby('weather')['total_delay'].mean()
    colors = {'clear': 'skyblue', 'rain': 'lightblue', 'snow': 'gray', 'fog': 'darkgray'}
    bar_colors = [colors.get(w, 'blue') for w in weather_delay.index]
    axes[1, 0].bar(range(len(weather_delay)), weather_delay.values, color=bar_colors)
    axes[1, 0].set_xticks(range(len(weather_delay)))
    axes[1, 0].set_xticklabels(weather_delay.index)
    axes[1, 0].set_xlabel('Weather Condition')
    axes[1, 0].set_ylabel('Average Delay (min)')
    axes[1, 0].set_title('Impact of Weather on Delay')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. PINN predictions vs actual
    pinn.eval()
    with torch.no_grad():
        sample_idx = np.random.choice(len(X_val), min(200, len(X_val)), replace=False)
        sample_X = torch.FloatTensor(X_val[sample_idx])
        predictions = pinn(sample_X)
        
        # Inverse transform
        predictions_actual = scaler_y.inverse_transform(predictions.numpy())
        y_actual = scaler_y.inverse_transform(y_val[sample_idx].reshape(-1, 1))
    
    axes[1, 1].scatter(y_actual, predictions_actual, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    axes[1, 1].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual Total Delay (min)')
    axes[1, 1].set_ylabel('Predicted Total Delay (min)')
    axes[1, 1].set_title('PINN Predictions vs Actual')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Calculate R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_actual, predictions_actual)
    axes[1, 1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 1].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Scenario comparison
    scenario_names = [r['scenario'] for r in results]
    total_delays = [r['total_delay'] for r in results]
    
    bars = axes[1, 2].bar(range(len(scenario_names)), total_delays, 
                         color=['red', 'orange', 'yellow', 'green', 'blue'])
    axes[1, 2].set_xticks(range(len(scenario_names)))
    axes[1, 2].set_xticklabels(scenario_names, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Total Network Delay (min)')
    axes[1, 2].set_title('Traffic Delay by Scenario')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, delay in zip(bars, total_delays):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{delay:.1f}', ha='center', va='bottom')
    
    plt.suptitle('Enhanced Traffic Circuit Analysis with PINN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('traffic_analysis_summary.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: traffic_analysis_summary.png")
    
    print("\n" + "=" * 60)
    print("8. RESULTS SUMMARY:")
    print("-" * 60)
    print(f"• Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"• Final validation loss: {history['val_loss'][-1]:.6f}")
    print(f"• Final physics loss: {history['physics_loss'][-1]:.6f}")
    print(f"• Learned BPR α parameter: {pinn.alpha.item():.4f}")
    print(f"• Learned BPR β parameter: {pinn.beta.item():.4f}")
    print(f"• Prediction R² score: {r2:.4f}")
    print(f"• Average network delay: {df['total_delay'].mean():.1f} min")
    print(f"• Peak congestion (Evening): {results[2]['congestion_index']:.3f}")
    print(f"• Worst scenario: {scenario_names[np.argmax(total_delays)]} "
          f"({max(total_delays):.1f} min delay)")
    
    print("\n✅ Analysis complete! All visualizations have been saved.")
    print("\n📊 Generated files:")
    print("   1. pinn_training_history.png - PINN training progress")
    print("   2. traffic_circuit_[scenario].png - Circuit diagrams for each scenario")
    print("   3. traffic_analysis_summary.png - Comprehensive analysis dashboard")

# ============================================================================
# 7. RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    main()