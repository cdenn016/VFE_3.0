"""Tiny CPU formula validation for proposed full-state receiver objectives."""
from __future__ import annotations
import json
from pathlib import Path
import torch

def spd(n: int) -> torch.Tensor:
    """Return a deterministic SPD covariance."""
    A = torch.randn(n, n, dtype=torch.float64, device="cpu")
    return A @ A.T + torch.eye(n, dtype=torch.float64)

def kl(
    mu_q:    torch.Tensor,  # (K,) receiver mean
    Sigma_q: torch.Tensor,  # (K, K) receiver covariance
    mu_p:    torch.Tensor,  # (K,) target mean
    Sigma_p: torch.Tensor,  # (K, K) target covariance
) -> torch.Tensor:
    r"""Compute full Gaussian KL(q || p)."""
    v = mu_q-mu_p
    return .5*(torch.trace(torch.linalg.solve(Sigma_p, Sigma_q))
               + v@torch.linalg.solve(Sigma_p,v)-mu_q.numel()
               + torch.logdet(Sigma_p)-torch.logdet(Sigma_q))

def main() -> None:
    """Check analytic forces, frozen coordinates, and outer gradients."""
    torch.set_num_threads(1)
    torch.manual_seed(53)
    K,H,J,d = 4,2,3,2
    mu = torch.randn(K,dtype=torch.float64,requires_grad=True)
    C = spd(K).requires_grad_()
    mp,Cp = torch.randn(K,dtype=torch.float64),spd(K)
    x = torch.randn(J,K,dtype=torch.float64)
    Cs = torch.stack([spd(K) for _ in range(J)])
    SQ,SK,T = [torch.randn(H,d,K,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    O = torch.randn(H,K,d,dtype=torch.float64,requires_grad=True)
    targets = [(O[h]@T[h]@x[j], .6*torch.eye(K,dtype=torch.float64)
                + O[h]@T[h]@Cs[j]@T[h].T@O[h].T) for h in range(H) for j in range(J)]
    keys = [(SK[h]@x[j],SK[h]@Cs[j]@SK[h].T) for h in range(H) for j in range(J)]
    a,gamma,lam,tau = .7,.4,.9,1.3
    pi = torch.full((H,J),1/J,dtype=torch.float64)
    def costs(m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Return comparison plus content costs."""
        return torch.stack([gamma*kl(SQ[h]@m,SQ[h]@S@SQ[h].T,*keys[h*J+j])
                            +lam*kl(m,S,*targets[h*J+j])
                            for h in range(H) for j in range(J)]).reshape(H,J)
    beta = torch.softmax(pi.log()-costs(mu,C)/tau,-1)
    w = beta.detach()
    def objective(m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Evaluate the frozen-label receiver objective."""
        return a*kl(m,S,mp,Cp)+(w*costs(m,S)).sum()
    gm,GC = torch.autograd.grad(objective(mu,C),(mu,C),create_graph=True)
    am = a*torch.linalg.solve(Cp,mu-mp)
    AC = .5*a*(torch.linalg.inv(Cp)-torch.linalg.inv(C))
    for h in range(H):
        A,V = SQ[h],SQ[h]@C@SQ[h].T
        for j in range(J):
            b,Vk = keys[h*J+j]
            m,R = targets[h*J+j]
            am = am+w[h,j]*(gamma*A.T@torch.linalg.solve(Vk,A@mu-b)+lam*torch.linalg.solve(R,mu-m))
            AC = AC+.5*w[h,j]*(gamma*(A.T@torch.linalg.solve(Vk,A)-A.T@torch.linalg.solve(V,A))
                               +lam*(torch.linalg.inv(R)-torch.linalg.inv(C)))
    dm,dC = torch.randn_like(mu),torch.randn_like(C)
    dC = .5*(dC+dC.T)
    eps = 1e-5
    fm = (objective(mu+eps*dm,C)-objective(mu-eps*dm,C))/(2*eps)
    fC = (objective(mu,C+eps*dC)-objective(mu,C-eps*dC))/(2*eps)
    reduced = a*kl(mu,C,mp,Cp)-tau*torch.logsumexp(pi.log()-costs(mu,C)/tau,-1).sum()
    em,eC = torch.autograd.grad(reduced,(mu,C),retain_graph=True)
    P,eta = a*torch.linalg.inv(Cp),a*torch.linalg.solve(Cp,mp)
    for h in range(H):
        for j in range(J):
            m,R = targets[h*J+j]
            P = P+lam*w[h,j]*torch.linalg.inv(R)
            eta = eta+lam*w[h,j]*torch.linalg.solve(R,m)
    mass = a+lam*w.sum()
    ms,Cstar = torch.linalg.solve(P,eta),mass*torch.linalg.inv(P)
    decoder = torch.randn(7,K,dtype=torch.float64)
    mn = mu-.01*(C@gm)
    ce = torch.nn.functional.cross_entropy((decoder@mn).unsqueeze(0),torch.tensor([2]))
    grads = torch.autograd.grad(ce,(SQ,SK,T,O))
    ce_det = torch.nn.functional.cross_entropy((decoder@mn.detach()).unsqueeze(0),torch.tensor([2]))
    bq = torch.softmax(pi.log()-torch.stack([
        gamma*kl(SQ[h]@mu,SQ[h]@C@SQ[h].T,*keys[h*J+j])
        for h in range(H) for j in range(J)]).reshape(H,J)/tau,-1)
    errors = {
        "mean_formula_max_error":(gm-am).abs().max().item(),
        "covariance_formula_max_error":(GC-AC).abs().max().item(),
        "mean_direction_fd_error":(fm-gm@dm).abs().item(),
        "covariance_direction_fd_error":(fC-(GC*dC).sum()).abs().item(),
        "envelope_mean_max_error":(em-gm).abs().max().item(),
        "envelope_covariance_max_error":(eC-GC).abs().max().item(),
        "content_optimum_mean_residual":(P@ms-eta).norm().item(),
        "content_optimum_covariance_residual":(.5*(P-mass*torch.linalg.inv(Cstar))).norm().item(),
    }
    result = {
        "scope":"K=4,H=2,J=3 formula validation only; CPU tensors, no production model",
        "torch_version":torch.__version__,"threads":torch.get_num_threads(),
        **errors,
        "exclusive_vs_inclusive_diagonal_max_gap":(mass/P.diag()-Cstar.diag()).abs().max().item(),
        "coupled_vs_qk_only_beta_max_gap":(beta-bq).abs().max().item(),
        "live_outer_gradient_norms":{n:g.norm().item() for n,g in zip(("S_Q","S_K","T","O"),grads)},
        "detached_ce_requires_grad":ce_det.requires_grad,
        "min_receiving_eigenvalue":min(torch.linalg.eigvalsh(R).min().item() for _,R in targets),
    }
    assert max(errors.values())<1e-7,errors
    assert all(g.norm().item()>0 for g in grads)
    assert not ce_det.requires_grad
    Path(__file__).with_name("derivation-checks.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
