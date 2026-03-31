import torch
import torch.nn as nn
from torchdiffeq import odeint


class SIRDynamics(nn.Module):
    """MLP that learns f(s, i, r, beta, gamma) -> (ds/dt, di/dt, dr/dt)."""

    def __init__(self, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = []
        in_dim = 5  # s, i, r, beta, gamma
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 3))  # ds, di, dr
        self.net = nn.Sequential(*layers)

    def forward(self, t, state):
        # state: (batch, 5) = [s, i, r, beta, gamma]
        return self.net(state)


class NeuralODE(nn.Module):
    """Parameter-conditioned Neural ODE for SIR dynamics."""

    def __init__(self, hidden_dim=64, n_layers=3, conservation_lambda=0.0):
        super().__init__()
        self.dynamics = SIRDynamics(hidden_dim=hidden_dim, n_layers=n_layers)
        self.conservation_lambda = conservation_lambda

    def forward(self, x0, params, t_grid):
        """
        Args:
            x0: (batch, 3) initial conditions [s0, i0, r0]
            params: (batch, 2) parameters [beta, gamma]
            t_grid: (T,) time points

        Returns:
            trajectories: (T, batch, 3)
        """
        full_state = torch.cat([x0, params], dim=-1)  # (batch, 5)

        def augmented_ode(t, y):
            dy = self.dynamics(t, y)
            # zero out the parameter derivatives (beta, gamma are constant)
            dy = torch.cat([dy[:, :3], torch.zeros_like(dy[:, :2])], dim=-1)
            return dy

        traj = odeint(augmented_ode, full_state, t_grid, method="dopri5",
                       rtol=1e-5, atol=1e-6)
        return traj[:, :, :3]  # (T, batch, 3) -- just s, i, r

    def get_derivatives(self, states, params):
        """Evaluate learned f(s, i, r, beta, gamma) at given points.

        Args:
            states: (N, 3) -- [s, i, r]
            params: (N, 2) -- [beta, gamma]

        Returns:
            derivatives: (N, 3) -- [ds/dt, di/dt, dr/dt]
        """
        x = torch.cat([states, params], dim=-1)
        with torch.no_grad():
            dy = self.dynamics(0.0, x)
        return dy[:, :3]

    def conservation_loss(self, derivatives):
        """Soft penalty: ds/dt + di/dt + dr/dt should be 0."""
        return (derivatives.sum(dim=-1) ** 2).mean()


def train_neural_ode(model, train_data, val_data, epochs=500, lr=1e-3,
                     weight_decay=0.01, batch_size=32, patience=50,
                     noise_sigma=0.005, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t_grid = torch.tensor(train_data["t_grid"], dtype=torch.float32, device=device)

    # prepare tensors
    train_s = torch.tensor(train_data["mean_s"], dtype=torch.float32)
    train_i = torch.tensor(train_data["mean_i"], dtype=torch.float32)
    train_r = torch.tensor(train_data["mean_r"], dtype=torch.float32)
    train_params = torch.tensor(train_data["params"][:, :2], dtype=torch.float32)
    n_train = len(train_params)

    val_s = torch.tensor(val_data["mean_s"], dtype=torch.float32, device=device)
    val_i = torch.tensor(val_data["mean_i"], dtype=torch.float32, device=device)
    val_r = torch.tensor(val_data["mean_r"], dtype=torch.float32, device=device)
    val_params = torch.tensor(val_data["params"][:, :2], dtype=torch.float32, device=device)
    val_targets = torch.stack([val_s, val_i, val_r], dim=-1)  # (N_val, T, 3)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]

            s = train_s[idx].to(device)
            i = train_i[idx].to(device)
            r = train_r[idx].to(device)
            p = train_params[idx].to(device)

            # add noise augmentation
            if noise_sigma > 0:
                s = s + torch.randn_like(s) * noise_sigma
                i = i + torch.randn_like(i) * noise_sigma
                r = r + torch.randn_like(r) * noise_sigma

            targets = torch.stack([s, i, r], dim=-1)  # (B, T, 3)
            x0 = targets[:, 0, :]  # (B, 3)

            pred = model(x0, p, t_grid)  # (T, B, 3)
            pred = pred.permute(1, 0, 2)  # (B, T, 3)

            loss = ((pred - targets) ** 2).mean()

            if model.conservation_lambda > 0:
                B_cur = targets.size(0)
                T_sub = targets[:, ::10, :].size(1)
                sample_states = targets[:, ::10, :].reshape(-1, 3)  # (B*T_sub, 3)
                sample_params = p.unsqueeze(1).expand(-1, T_sub, -1).reshape(-1, 2)  # (B*T_sub, 2)
                derivs = model.get_derivatives(sample_states, sample_params)
                loss = loss + model.conservation_lambda * model.conservation_loss(derivs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(1, n_batches)
        history["train_loss"].append(avg_train)

        # validate
        model.eval()
        with torch.no_grad():
            val_x0 = val_targets[:, 0, :]
            val_pred = model(val_x0, val_params, t_grid).permute(1, 0, 2)
            val_loss = ((val_pred - val_targets) ** 2).mean().item()
        history["val_loss"].append(val_loss)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} -- train: {avg_train:.6f}, val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
