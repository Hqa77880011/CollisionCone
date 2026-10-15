import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm


METHODS = [
    "collisioncone",
    "always_metric",
    "no_ego_compensation",
    "cone_only",
    "no_nuisance_marginalization",
]


def observations(state, times, ego):
    r, closing, beta, beta_rate, log_size = state
    direction = np.array([np.cos(beta), np.sin(beta)])
    tangent = np.array([-np.sin(beta), np.cos(beta)])
    p = r * direction
    v = closing * direction + r * beta_rate * tangent
    relative = p[None] + times[:, None] * v[None] - ego
    distance = np.linalg.norm(relative, axis=1)
    return np.column_stack(
        [np.arctan2(relative[:, 1], relative[:, 0]), log_size - np.log(np.maximum(distance, 1e-8))]
    )


def derivative(function, state):
    columns = []
    for i in range(len(state)):
        step = 1e-5 * max(1, abs(state[i]))
        plus, minus = state.copy(), state.copy()
        plus[i] += step
        minus[i] -= step
        columns.append(((function(plus) - function(minus)) / (2 * step)).ravel())
    return np.column_stack(columns)


def generate(rng, count):
    for index in range(count):
        times = np.linspace(-3, 0, 16)
        state = np.array(
            [
                rng.uniform(20, 100),
                rng.uniform(-5, -1),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.01, 0.01),
                np.log(rng.uniform(1, 3)),
            ]
        )
        ego = np.column_stack(
            [np.zeros(len(times)), 0.2 * times**2 if index % 2 else np.zeros(len(times))]
        )
        yaw = 0.03 * np.sin(times)
        noiseless = observations(state, times, ego)
        measured = noiseless + np.array([[rng.gauss(0, 0.001), rng.gauss(0, 0.005)] for _ in times])
        measured[:, 0] -= yaw
        yield {
            "id": str(index),
            "family": f"encounter-{index}",
            "times": times.tolist(),
            "measurements": measured.tolist(),
            "ego_positions": ego.tolist(),
            "ego_yaw": yaw.tolist(),
            "noise_std": [0.001, 0.005],
            "initial": [50.0, -2.0, 0.0, 0.0, 0.7],
            "prior_std": [100.0, 10.0, 0.5, 0.1, 2.0],
            "information_scale": [1.0, 1.0],
            "observability_threshold": 0.001,
            "bearing_rate_limit": 0.01,
            "expansion_limit": 0.005,
            "warning_threshold": 0.5,
            "truth": state.tolist(),
            "risk_horizon": 30.0,
            "duration_hours": 4 / 3600,
            "seed": rng.randrange(2**31),
        }


def run_case(case, method):
    times = np.asarray(case["times"])
    ego = np.asarray(case["ego_positions"])
    measured = np.asarray(case["measurements"]).copy()
    if method != "no_ego_compensation":
        measured[:, 0] += np.asarray(case["ego_yaw"])
    standard = np.tile(np.asarray(case["noise_std"]), len(times))
    initial = np.asarray(case["initial"], dtype=float)

    def residual(state):
        error = observations(state, times, ego) - measured
        error[:, 0] = np.arctan2(np.sin(error[:, 0]), np.cos(error[:, 0]))
        return error.ravel() / standard

    fitted = least_squares(
        residual,
        initial,
        bounds=([0.1, -30, -np.pi, -0.5, -5], [1000, 30, np.pi, 0.5, 5]),
        max_nfev=1000,
    )
    if not fitted.success:
        raise RuntimeError("Relative-state fit failed: " + fitted.message)
    J = derivative(lambda state: observations(state, times, ego), fitted.x) / standard[:, None]
    information = J.T @ J
    if method == "no_nuisance_marginalization":
        metric_info = information[:2, :2]
    else:
        metric_info = (
            information[:2, :2]
            - information[:2, 2:] @ np.linalg.pinv(information[2:, 2:]) @ information[2:, :2]
        )
    scale = np.diag(1 / np.sqrt(case["information_scale"]))
    score = float(np.linalg.eigvalsh(scale @ metric_info @ scale)[0])
    design = np.column_stack([np.ones(len(times)), times])
    slope_cov = np.linalg.inv(design.T @ design)[1, 1]
    slopes = np.linalg.lstsq(design, measured, rcond=None)[0][1]
    bearing_sigma = np.sqrt(slope_cov) * case["noise_std"][0]
    scale_sigma = np.sqrt(slope_cov) * case["noise_std"][1]
    cone = (
        norm.cdf((case["bearing_rate_limit"] - slopes[0]) / bearing_sigma)
        - norm.cdf((-case["bearing_rate_limit"] - slopes[0]) / bearing_sigma)
    ) * norm.sf((case["expansion_limit"] - slopes[1]) / scale_sigma)
    metric = (
        score >= case["observability_threshold"] and method != "cone_only"
    ) or method == "always_metric"
    result = {
        "observability": score,
        "metric_censored": not metric,
        "cone_probability": float(cone),
        "ttc_interval": None,
        "cpa_interval": None,
        "risk": None,
    }
    if metric:
        prior = np.diag(1 / np.asarray(case["prior_std"]) ** 2)
        covariance = np.linalg.pinv(information + prior)
        samples = np.random.default_rng(case["seed"]).multivariate_normal(
            fitted.x, covariance, size=2000
        )
        range_, closing, beta, beta_rate = (
            samples[:, 0],
            samples[:, 1],
            samples[:, 2],
            samples[:, 3],
        )
        approaching = (closing < 0) & (range_ > 0)
        ttc = -range_[approaching] / closing[approaching]
        positions = range_[:, None] * np.column_stack([np.cos(beta), np.sin(beta)])
        velocity = closing[:, None] * np.column_stack([np.cos(beta), np.sin(beta)]) + (
            range_ * beta_rate
        )[:, None] * np.column_stack([-np.sin(beta), np.cos(beta)])
        time_cpa = np.maximum(
            0, -(positions * velocity).sum(1) / np.maximum((velocity**2).sum(1), 1e-12)
        )
        distance = np.linalg.norm(positions + time_cpa[:, None] * velocity, axis=1)
        result["ttc_interval"] = np.quantile(ttc, [0.05, 0.95]).tolist() if len(ttc) else None
        result["cpa_interval"] = np.quantile(distance, [0.05, 0.95]).tolist()
        result["risk"] = float(
            np.mean(approaching & (-range_ / np.minimum(closing, -1e-8) < case["risk_horizon"]))
        )
    return result


def score(case, result):
    state = case["truth"]
    truth_ttc = -state[0] / state[1] if state[1] < 0 else None
    high_risk = (
        truth_ttc is not None
        and truth_ttc < case["risk_horizon"]
        and abs(state[3]) <= case["bearing_rate_limit"]
    )
    probability = result["risk"] if result["risk"] is not None else result["cone_probability"]
    warning = probability >= case["warning_threshold"]
    interval = result["ttc_interval"]
    return {
        "high_risk_recall": int(warning) if high_risk else None,
        "false_alarms": int(warning and not high_risk),
        "duration_hours": case["duration_hours"],
        "warning_lead_seconds": truth_ttc if warning and high_risk else None,
        "false_precision": int(not result["metric_censored"])
        if result["observability"] < case["observability_threshold"]
        else None,
        "metric_censoring": int(result["metric_censored"]),
        "ttc_coverage90": int(interval[0] <= truth_ttc <= interval[1])
        if interval and truth_ttc is not None
        else None,
        "brier": float((probability - high_risk) ** 2),
        "bernoulli_nll": float(-np.log(max(probability if high_risk else 1 - probability, 1e-12))),
        "legal_effects": int(warning == high_risk),
    }
