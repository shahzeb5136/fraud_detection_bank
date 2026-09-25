"""Hierarchical Bayesian fraud-rate models fitted by MCMC.

Two models, both sampled with hand-written, fully traceable NumPy samplers (no compiled back-end
needed), four chains run in parallel as a leading array dimension:

1. Merchant hierarchy  -  vendor within MCC within population (the pooling structure requested in
   the data dictionary for `mcc`: "back-off level for unseen vendors").
       y_v ~ Binomial(n_v, sigmoid(b_v))
       b_v ~ Normal(a_{m(v)}, sigma_v^2)
       a_m ~ Normal(mu, sigma_m^2)
       mu  ~ Normal(logit(0.01), 1.5^2),   sigma_m, sigma_v ~ HalfNormal(1)
   Sampler: Metropolis-within-Gibbs. Vendor effects use a vectorised random-walk Metropolis step
   (conditionally independent given the MCC effects), MCC effects and mu are exact Gibbs draws,
   the two scale parameters use adaptive random-walk Metropolis on the log scale.

2. Customer propensity  -  Beta-Binomial hierarchy.
       y_p ~ Binomial(n_p, theta_p),  theta_p ~ Beta(kappa*phi, kappa*(1-phi))
       phi ~ Beta(2, 200),  log kappa ~ Normal(log 50, 1)
   theta_p is integrated out analytically; (logit phi, log kappa) are sampled with an adaptive 2-D
   random-walk Metropolis. Per-customer posteriors are then conjugate Beta draws.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, logit

from . import config as C


# ----------------------------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------------------------
def split_rhat(x: np.ndarray) -> float:
    """x: (chains, draws). Split-R-hat (Gelman et al. 2013 / Vehtari et al. 2021)."""
    c, n = x.shape
    half = n // 2
    parts = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)
    m, n2 = parts.shape
    chain_means = parts.mean(axis=1)
    chain_vars = parts.var(axis=1, ddof=1)
    B = n2 * chain_means.var(ddof=1)
    W = chain_vars.mean()
    var_hat = (n2 - 1) / n2 * W + B / n2
    return float(np.sqrt(var_hat / W)) if W > 0 else np.nan


def ess(x: np.ndarray) -> float:
    """Effective sample size from chain-averaged autocorrelation with Geyer's initial positive sequence."""
    c, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    acov = np.zeros(n)
    for ch in range(c):
        f = np.fft.rfft(np.concatenate([xc[ch], np.zeros(n)]))
        ac = np.fft.irfft(f * np.conj(f))[:n] / n
        acov += ac
    acov /= c
    if acov[0] <= 0:
        return float(c * n)
    rho = acov / acov[0]
    # Geyer: sum pairs while positive
    s, t = 0.0, 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
        t += 2
    tau = 1 + 2 * s
    return float(c * n / max(tau, 1e-9))


def summarize(draws: np.ndarray, name: str) -> dict:
    flat = draws.reshape(-1)
    return dict(parameter=name, mean=float(flat.mean()), sd=float(flat.std(ddof=1)),
                q2_5=float(np.quantile(flat, 0.025)), q50=float(np.quantile(flat, 0.5)), q97_5=float(np.quantile(flat, 0.975)),
                rhat=split_rhat(draws), ess=ess(draws))


# ----------------------------------------------------------------------------------------------
# Model 1: vendor within MCC within population
# ----------------------------------------------------------------------------------------------
class VendorHierarchyMCMC:
    def __init__(self, vendor_ids, n_v, y_v, mcc_of_vendor, mcc_ids, chains=4, iters=7000, warmup=2500, seed=C.SEED, thin_keep=400):
        self.vendor_ids = np.asarray(vendor_ids)
        self.n = np.asarray(n_v, dtype=float)
        self.y = np.asarray(y_v, dtype=float)
        self.mcc_ids = list(mcc_ids)
        self.mcc_index = {m: i for i, m in enumerate(self.mcc_ids)}
        self.midx = np.array([self.mcc_index[m] for m in mcc_of_vendor])
        self.V, self.M = len(self.n), len(self.mcc_ids)
        self.chains, self.iters, self.warmup = chains, iters, warmup
        self.rng = np.random.default_rng(seed)
        self.thin_keep = thin_keep
        self.mu0, self.s0 = float(logit(0.01)), 1.5
        self.ind = np.zeros((self.V, self.M))
        self.ind[np.arange(self.V), self.midx] = 1.0
        self.n_per_m = self.ind.sum(axis=0)

    # log-likelihood pieces --------------------------------------------------------------
    def _ll_b(self, b, a, sv):
        return self.y * b - self.n * np.logaddexp(0.0, b) - 0.5 * ((b - a[:, self.midx]) / sv[:, None]) ** 2

    def _lp_sigma(self, log_s, resid_sq_sum, count):
        s = np.exp(log_s)
        return -count * log_s - 0.5 * resid_sq_sum / s ** 2 - 0.5 * (s / 1.0) ** 2 + log_s   # HalfNormal(1) prior + Jacobian

    def run(self, log=print):
        rng, Ch, V, M = self.rng, self.chains, self.V, self.M
        raw = np.log((self.y + 0.5) / (self.n - self.y + 0.5))
        b = np.tile(raw, (Ch, 1)) + rng.normal(0, 0.3, (Ch, V))
        a = np.stack([np.array([raw[self.midx == m].mean() for m in range(M)]) for _ in range(Ch)]) + rng.normal(0, 0.3, (Ch, M))
        mu = a.mean(axis=1) + rng.normal(0, 0.2, Ch)
        log_sm = np.full(Ch, np.log(0.5)) + rng.normal(0, 0.2, Ch)
        log_sv = np.full(Ch, np.log(0.5)) + rng.normal(0, 0.2, Ch)
        step_b = np.full((Ch, V), 0.5)
        step_sm = np.full(Ch, 0.3)
        step_sv = np.full(Ch, 0.3)
        keep = self.iters - self.warmup
        D = dict(mu=np.zeros((Ch, keep)), sigma_m=np.zeros((Ch, keep)), sigma_v=np.zeros((Ch, keep)), a=np.zeros((Ch, keep, M)))
        thin_idx = set(np.linspace(0, keep - 1, min(self.thin_keep, keep)).astype(int).tolist())
        b_thin = []
        b_sum = np.zeros((Ch, V)); b_sq = np.zeros((Ch, V))
        acc_b = np.zeros((Ch, V)); acc_sm = np.zeros(Ch); acc_sv = np.zeros(Ch)
        for it in range(self.iters):
            sv = np.exp(log_sv); sm = np.exp(log_sm)
            # --- vendor effects: vectorised Metropolis --------------------------------------
            prop = b + step_b * rng.normal(size=(Ch, V))
            d = self._ll_b(prop, a, sv) - self._ll_b(b, a, sv)
            acc = np.log(rng.random((Ch, V))) < d
            b = np.where(acc, prop, b)
            # --- MCC effects: Gibbs --------------------------------------------------------
            prec = 1.0 / sm[:, None] ** 2 + self.n_per_m[None, :] / sv[:, None] ** 2
            mean = (mu[:, None] / sm[:, None] ** 2 + (b @ self.ind) / sv[:, None] ** 2) / prec
            a = mean + rng.normal(size=(Ch, M)) / np.sqrt(prec)
            # --- population mean: Gibbs ---------------------------------------------------------
            prec_mu = 1.0 / self.s0 ** 2 + M / sm ** 2
            mean_mu = (self.mu0 / self.s0 ** 2 + a.sum(axis=1) / sm ** 2) / prec_mu
            mu = mean_mu + rng.normal(size=Ch) / np.sqrt(prec_mu)
            # --- scales: Metropolis on log scale ------------------------------------------------
            rss_m = ((a - mu[:, None]) ** 2).sum(axis=1)
            prop = log_sm + step_sm * rng.normal(size=Ch)
            d = self._lp_sigma(prop, rss_m, M) - self._lp_sigma(log_sm, rss_m, M)
            ok = np.log(rng.random(Ch)) < d
            log_sm = np.where(ok, prop, log_sm)
            rss_v = ((b - a[:, self.midx]) ** 2).sum(axis=1)
            prop = log_sv + step_sv * rng.normal(size=Ch)
            d = self._lp_sigma(prop, rss_v, V) - self._lp_sigma(log_sv, rss_v, V)
            ok2 = np.log(rng.random(Ch)) < d
            log_sv = np.where(ok2, prop, log_sv)
            # --- adaptation during warm-up ---------------------------------------------------
            if it < self.warmup:
                if it % 50 == 49:
                    rate = acc_b / 50.0
                    step_b *= np.where(rate < 0.35, 0.8, np.where(rate > 0.55, 1.25, 1.0))
                    step_sm *= np.where(acc_sm / 50.0 < 0.3, 0.8, np.where(acc_sm / 50.0 > 0.5, 1.25, 1.0))
                    step_sv *= np.where(acc_sv / 50.0 < 0.3, 0.8, np.where(acc_sv / 50.0 > 0.5, 1.25, 1.0))
                    acc_b[:] = 0; acc_sm[:] = 0; acc_sv[:] = 0
                else:
                    acc_b += acc; acc_sm += ok; acc_sv += ok2
            else:
                k = it - self.warmup
                acc_b += acc; acc_sm += ok; acc_sv += ok2
                D["mu"][:, k] = mu; D["sigma_m"][:, k] = sm; D["sigma_v"][:, k] = sv; D["a"][:, k, :] = a
                b_sum += b; b_sq += b * b
                if k in thin_idx:
                    b_thin.append(b.copy())
            if it % 1000 == 999:
                log(f"    MCMC iteration {it + 1}/{self.iters}")
        b_thin = np.stack(b_thin)                       # (T, Ch, V)
        rate_draws = expit(b_thin).reshape(-1, V)      # (T*Ch, V)
        self.draws = D
        self.b_thin = b_thin
        self.acceptance = dict(vendor=float((acc_b / keep).mean()), sigma_m=float((acc_sm / keep).mean()), sigma_v=float((acc_sv / keep).mean()))
        self.vendor_summary = dict(
            vendor_id=self.vendor_ids, n=self.n, y=self.y, raw_rate=self.y / np.maximum(self.n, 1),
            post_mean=rate_draws.mean(axis=0), post_sd=rate_draws.std(axis=0),
            q2_5=np.quantile(rate_draws, 0.025, axis=0), q97_5=np.quantile(rate_draws, 0.975, axis=0),
            prob_gt_3x_pop=(rate_draws > 3 * expit(D["mu"]).mean()).mean(axis=0),
            mcc=np.array([self.mcc_ids[i] for i in self.midx]),
        )
        a_flat = D["a"].reshape(-1, M)
        self.mcc_summary = dict(mcc=np.array(self.mcc_ids), post_mean=expit(a_flat).mean(axis=0),
                                q2_5=np.quantile(expit(a_flat), 0.025, axis=0), q97_5=np.quantile(expit(a_flat), 0.975, axis=0),
                                n_vendors=self.n_per_m, n=np.array([self.n[self.midx == i].sum() for i in range(M)]),
                                y=np.array([self.y[self.midx == i].sum() for i in range(M)]))
        self.hyper_summary = [summarize(D["mu"], "mu (population log-odds)"), summarize(D["sigma_m"], "sigma_mcc"), summarize(D["sigma_v"], "sigma_vendor")] + \
                             [summarize(D["a"][:, :, i], f"a[{m}]") for i, m in enumerate(self.mcc_ids)]
        self.pop_rate = float(expit(D["mu"]).mean())
        self.vendor_pos = {v: i for i, v in enumerate(self.vendor_ids)}
        log(f"    done: acceptance vendor={self.acceptance['vendor']:.2f} sigma_m={self.acceptance['sigma_m']:.2f} sigma_v={self.acceptance['sigma_v']:.2f}")
        return self

    # predictions ------------------------------------------------------------------------------
    def vendor_rate(self, vendor_id: str, mcc: str | None = None) -> dict:
        """Posterior fraud rate for a merchant; unseen merchants back off to their MCC, unseen MCCs to the population."""
        if vendor_id in self.vendor_pos:
            i = self.vendor_pos[vendor_id]
            s = self.vendor_summary
            return dict(level="vendor", mean=float(s["post_mean"][i]), lo=float(s["q2_5"][i]), hi=float(s["q97_5"][i]),
                        n=int(s["n"][i]), y=int(s["y"][i]), raw=float(s["raw_rate"][i]), prob_gt_3x=float(s["prob_gt_3x_pop"][i]))
        return self.new_vendor_rate(mcc)

    def new_vendor_rate(self, mcc: str | None) -> dict:
        D = self.draws
        rng = np.random.default_rng(0)
        sv = D["sigma_v"].reshape(-1)
        if mcc in self.mcc_index:
            a = D["a"][:, :, self.mcc_index[mcc]].reshape(-1)
            level = "mcc (unseen merchant)"
        else:
            a = D["mu"].reshape(-1) + D["sigma_m"].reshape(-1) * rng.normal(size=sv.shape)
            level = "population (unseen MCC)"
        draws = expit(a + sv * rng.normal(size=sv.shape))
        return dict(level=level, mean=float(draws.mean()), lo=float(np.quantile(draws, 0.025)), hi=float(np.quantile(draws, 0.975)),
                    n=0, y=0, raw=np.nan, prob_gt_3x=float((draws > 3 * self.pop_rate).mean()))

    def vendor_draws(self, vendor_id: str) -> np.ndarray:
        i = self.vendor_pos[vendor_id]
        return expit(self.b_thin[:, :, i].reshape(-1))


# ----------------------------------------------------------------------------------------------
# Model 2: customer propensity (Beta-Binomial)
# ----------------------------------------------------------------------------------------------
class PersonPropensityMCMC:
    def __init__(self, person_ids, n_p, y_p, chains=4, iters=6000, warmup=2000, seed=C.SEED + 1):
        self.person_ids = np.asarray(person_ids)
        self.n = np.asarray(n_p, dtype=float)
        self.y = np.asarray(y_p, dtype=float)
        self.chains, self.iters, self.warmup = chains, iters, warmup
        self.rng = np.random.default_rng(seed)

    def _logpost(self, lphi, lkap):
        phi, kap = expit(lphi), np.exp(lkap)
        al, be = kap * phi, kap * (1 - phi)
        y, n = self.y[None, :], self.n[None, :]
        ll = (gammaln(y + al[:, None]) + gammaln(n - y + be[:, None]) - gammaln(n + al[:, None] + be[:, None])
              + gammaln(al[:, None] + be[:, None]) - gammaln(al[:, None]) - gammaln(be[:, None])).sum(axis=1)
        lprior = (2 - 1) * np.log(phi) + (200 - 1) * np.log(1 - phi) + np.log(phi) + np.log(1 - phi)        # Beta(2,200) + Jacobian
        lprior += -0.5 * ((lkap - np.log(50)) / 1.0) ** 2                                                    # LogNormal(log 50, 1) on kappa (Jacobian absorbed: sampling log kappa)
        return ll + lprior

    def run(self, log=print):
        rng, Ch = self.rng, self.chains
        lphi = np.full(Ch, float(logit(0.01))) + rng.normal(0, 0.3, Ch)
        lkap = np.full(Ch, np.log(40.0)) + rng.normal(0, 0.3, Ch)
        step = np.full((Ch, 2), 0.3)
        cur = self._logpost(lphi, lkap)
        keep = self.iters - self.warmup
        D = dict(phi=np.zeros((Ch, keep)), kappa=np.zeros((Ch, keep)))
        acc = np.zeros(Ch)
        acc_tot = np.zeros(Ch)
        for it in range(self.iters):
            z = rng.normal(size=(Ch, 2)) * step
            p_lphi, p_lkap = lphi + z[:, 0], lkap + z[:, 1]
            new = self._logpost(p_lphi, p_lkap)
            ok = np.log(rng.random(Ch)) < new - cur
            lphi = np.where(ok, p_lphi, lphi); lkap = np.where(ok, p_lkap, lkap); cur = np.where(ok, new, cur)
            if it < self.warmup:
                acc += ok
                if it % 50 == 49:
                    r = acc / 50
                    step *= np.where(r < 0.2, 0.7, np.where(r > 0.4, 1.3, 1.0))[:, None]
                    acc[:] = 0
            else:
                k = it - self.warmup
                acc_tot += ok
                D["phi"][:, k] = expit(lphi); D["kappa"][:, k] = np.exp(lkap)
        self.draws = D
        self.acceptance = float((acc_tot / keep).mean())
        self.hyper_summary = [summarize(D["phi"], "phi (population propensity)"), summarize(D["kappa"], "kappa (concentration)")]
        # per-customer posterior via 200 thinned hyper-draws
        idx = np.linspace(0, keep - 1, 200).astype(int)
        phi, kap = D["phi"][:, idx].reshape(-1), D["kappa"][:, idx].reshape(-1)
        al = kap * phi; be = kap * (1 - phi)
        post_mean = ((al[:, None] + self.y[None, :]) / (al[:, None] + be[:, None] + self.n[None, :]))
        theta = rng.beta(al[:, None] + self.y[None, :], be[:, None] + self.n[None, :] - self.y[None, :])
        self.person_summary = dict(person_id=self.person_ids, n=self.n, y=self.y, raw_rate=self.y / np.maximum(self.n, 1),
                                   post_mean=post_mean.mean(axis=0), q2_5=np.quantile(theta, 0.025, axis=0), q97_5=np.quantile(theta, 0.975, axis=0))
        self.person_pos = {p: i for i, p in enumerate(self.person_ids)}
        self.pop_mean = float(D["phi"].mean())
        self.alpha_beta = (float(al.mean()), float(be.mean()))
        log(f"    done: acceptance={self.acceptance:.2f}, phi={self.pop_mean:.4f}, kappa={float(D['kappa'].mean()):.1f}")
        return self

    def person_rate(self, person_id: str) -> dict:
        if person_id in self.person_pos:
            i = self.person_pos[person_id]
            s = self.person_summary
            return dict(level="customer", mean=float(s["post_mean"][i]), lo=float(s["q2_5"][i]), hi=float(s["q97_5"][i]), n=int(s["n"][i]), y=int(s["y"][i]))
        al, be = self.alpha_beta
        rng = np.random.default_rng(0)
        th = rng.beta(al, be, 4000)
        return dict(level="population (new customer)", mean=self.pop_mean, lo=float(np.quantile(th, 0.025)), hi=float(np.quantile(th, 0.975)), n=0, y=0)


def channel_posteriors(df) -> list[dict]:
    """Conjugate Beta(1,1) posteriors of the fraud rate per channel (analytic, no sampling needed)."""
    out = []
    for ch, g in df.groupby("channel"):
        n, y = len(g), int(g.fraud_label.sum())
        a, b = 1 + y, 1 + n - y
        from scipy.stats import beta as _beta
        out.append(dict(channel=ch, n=n, y=y, post_mean=a / (a + b), q2_5=float(_beta.ppf(0.025, a, b)), q97_5=float(_beta.ppf(0.975, a, b))))
    return out
