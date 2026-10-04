"""Minimal Fourier Neural Operator (Li et al., 2021) baseline for honest out-of-sample comparisons.

Trained for next-step prediction u(t) -> u(t + dt) on the SAME (noisy) training data the discovery methods see, then
rolled out autoregressively. 1-D and 2-D periodic grids, any number of fields.

    model = train_fno(U_train, epochs=..., modes=..., width=...)   # U_train: (n_traj, nt, *grid, n_fields)
    Y = rollout_fno(model, U0, n_steps)                            # -> (n_steps + 1, *grid, n_fields)
"""
import numpy as np


def _torch():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch, dev


def make_fno(ndim, n_fields, modes=16, width=48, layers=4):
    torch, _ = _torch()
    nn = torch.nn

    class SpectralConv(nn.Module):
        def __init__(self, c, m):
            super().__init__()
            self.m = m
            scale = 1 / (c * c)
            shape = (c, c, m) if ndim == 1 else (c, c, m, m)
            self.wr = nn.Parameter(scale * torch.randn(*shape))
            self.wi = nn.Parameter(scale * torch.randn(*shape))
            if ndim == 2:
                self.wr2 = nn.Parameter(scale * torch.randn(*shape))
                self.wi2 = nn.Parameter(scale * torch.randn(*shape))

        def forward(self, x):
            if ndim == 1:
                xf = torch.fft.rfft(x.float().cpu()) if x.device.type == "mps" else torch.fft.rfft(x)
                xf = xf.to(x.device) if x.device.type != "mps" else xf
                out = torch.zeros(x.shape[0], x.shape[1], xf.shape[-1], dtype=torch.cfloat, device=xf.device)
                w = torch.complex(self.wr, self.wi).to(xf.device)
                out[..., :self.m] = torch.einsum("bix,iox->box", xf[..., :self.m], w)
                y = torch.fft.irfft(out, n=x.shape[-1])
                return y.to(x.device)
            xf = torch.fft.rfft2(x.float().cpu()) if x.device.type == "mps" else torch.fft.rfft2(x)
            out = torch.zeros(x.shape[0], x.shape[1], x.shape[-2], xf.shape[-1], dtype=torch.cfloat, device=xf.device)
            w1 = torch.complex(self.wr, self.wi).to(xf.device)
            w2 = torch.complex(self.wr2, self.wi2).to(xf.device)
            out[..., :self.m, :self.m] = torch.einsum("bixy,ioxy->boxy", xf[..., :self.m, :self.m], w1)
            out[..., -self.m:, :self.m] = torch.einsum("bixy,ioxy->boxy", xf[..., -self.m:, :self.m], w2)
            y = torch.fft.irfft2(out, s=x.shape[-2:])
            return y.to(x.device)

    class FNO(nn.Module):
        def __init__(self):
            super().__init__()
            conv = nn.Conv1d if ndim == 1 else nn.Conv2d
            self.lift = conv(n_fields, width, 1)
            self.spec = nn.ModuleList([SpectralConv(width, modes) for _ in range(layers)])
            self.loc = nn.ModuleList([conv(width, width, 1) for _ in range(layers)])
            self.proj1 = conv(width, 128, 1)
            self.proj2 = conv(128, n_fields, 1)

        def forward(self, x):                      # x: (b, fields, *grid) -> residual update
            h = self.lift(x)
            for s, l in zip(self.spec, self.loc):
                h = torch.nn.functional.gelu(s(h) + l(h))
            return x + self.proj2(torch.nn.functional.gelu(self.proj1(h)))
    return FNO()


def train_fno(U, epochs=300, modes=16, width=48, layers=4, lr=2e-3, batch=32, seed=0, verbose=False, max_minutes=10):
    """U: (n_traj, nt, *grid, n_fields) training data (already noisy if the scenario is noisy)."""
    import time
    torch, dev = _torch()
    torch.manual_seed(seed)
    ndim = U.ndim - 3
    nf = U.shape[-1]
    mu, sd = U.reshape(-1, nf).mean(0), U.reshape(-1, nf).std(0) + 1e-12
    Z = (U - mu) / sd
    X = np.concatenate([Z[j, :-1] for j in range(Z.shape[0])])
    Y = np.concatenate([Z[j, 1:] for j in range(Z.shape[0])])
    perm = (0, ndim + 1) + tuple(range(1, ndim + 1))           # (n, *grid, f) -> (n, f, *grid)
    X = torch.tensor(np.transpose(X, perm), dtype=torch.float32, device=dev)
    Y = torch.tensor(np.transpose(Y, perm), dtype=torch.float32, device=dev)
    model = make_fno(ndim, nf, modes, width, layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randperm(len(X), device=dev)
        tot = 0.0
        for b in range(0, len(X), batch):
            i = idx[b:b + batch]
            loss = torch.mean((model(X[i]) - Y[i]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(i)
        sched.step()
        if verbose and ep % 25 == 0:
            print(f"  fno epoch {ep} loss {tot / len(X):.3e}", flush=True)
        if time.time() - t0 > max_minutes * 60:
            break
    model.eval()
    model.norm = (mu, sd, perm)
    return model


def rollout_fno(model, U0, n_steps):
    torch, dev = _torch()
    mu, sd, perm = model.norm
    ndim = U0.ndim - 1
    inv = (0,) + tuple(range(2, ndim + 2)) + (1,)
    x = torch.tensor(np.transpose(((U0 - mu) / sd)[None], perm), dtype=torch.float32, device=dev)
    out = [U0]
    with torch.no_grad():
        for _ in range(n_steps):
            x = model(x)
            out.append(np.transpose(x.cpu().numpy(), inv)[0] * sd + mu)
    return np.array(out)


# ----------------------------------------------------------------------------- ODEs: FNO over a time window
def train_fno_window(U, W=16, epochs=400, modes=8, width=48, layers=4, lr=2e-3, batch=256, seed=0, val_frac=0.2,
                     max_minutes=5, verbose=False):
    """ODE data (no spatial grid): the FNO's 1-D domain is a window of W time steps. It maps the last W states to
    the next W (Li et al.'s time-window variant). U: (n_traj, nt, n_vars), as given (noise, spikes, gaps-free).
    The last val_frac of every trajectory is held out; the checkpoint with the lowest held-out loss is kept."""
    import copy
    import time
    torch, _ = _torch()
    dev = "cpu"                                         # tiny model: CPU avoids the MPS FFT round-trips
    torch.manual_seed(seed)
    nf = U.shape[-1]
    mu, sd = U.reshape(-1, nf).mean(0), U.reshape(-1, nf).std(0) + 1e-12
    Z = (U - mu) / sd

    def pairs(seq):
        n = len(seq) - 2 * W + 1
        if n <= 0:
            return np.zeros((0, nf, W)), np.zeros((0, nf, W))
        X = np.stack([seq[i:i + W].T for i in range(n)])
        return X, np.stack([seq[i + W:i + 2 * W].T for i in range(n)])
    k = int(Z.shape[1] * (1 - val_frac))
    tr = [pairs(Z[j, :k]) for j in range(Z.shape[0])]
    va = [pairs(Z[j, k - W:]) for j in range(Z.shape[0])]
    t_ = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float32, device=dev)
    Xt, Yt = t_([p[0] for p in tr]), t_([p[1] for p in tr])
    Xv, Yv = t_([p[0] for p in va]), t_([p[1] for p in va])
    model = make_fno(1, nf, min(modes, W // 2 + 1), width, layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, t0 = np.inf, None, time.time()
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for b in range(0, len(Xt), batch):
            i = idx[b:b + batch]
            loss = torch.mean((model(Xt[i]) - Yt[i]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(torch.mean((model(Xv) - Yv) ** 2))
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
        if verbose and ep % 25 == 0:
            print(f"  fno-window epoch {ep} val {v:.3e}", flush=True)
        if time.time() - t0 > max_minutes * 60:
            break
    model.load_state_dict(best_state)
    model.eval()
    model.norm = (mu, sd, W)
    model.info = {"epochs": ep + 1, "val_loss": best, "n_train_windows": len(Xt), "minutes": (time.time() - t0) / 60}
    return model


def rollout_fno_window(model, U_start, n_total):
    """U_start: the first W states (n_vars last). Returns (n_total, n_vars): U_start, then chained window forecasts."""
    torch, _ = _torch()
    mu, sd, W = model.norm
    z = ((np.asarray(U_start[:W]) - mu) / sd).T[None]
    out = [np.asarray(U_start[:W])]
    x = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        while sum(len(o) for o in out) < n_total:
            x = model(x)
            if not torch.isfinite(x).all():
                out.append(np.full((W, len(mu)), np.nan))
                x = torch.nan_to_num(x)
                continue
            out.append(x[0].numpy().T * sd + mu)
    return np.concatenate(out)[:n_total]
