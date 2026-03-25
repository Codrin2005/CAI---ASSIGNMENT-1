from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import pstdev
from typing import Any

from negmas.sao import SAONegotiator, SAOResponse, SAOState, ResponseType

class Group34_Negotiator_BOA(SAONegotiator):
    """
    BOA-style negotiator with:
    - time-dependent Boulware-like concession
    - lightweight frequency-based opponent model
    - issue-importance-aware bid construction
    - opponent first-offer anchoring
    - opponent concession tracking
    - late-stage reuse of opponent bid history
    - forward-looking acceptance
    """

    def __init__(
        self,
        *args,
        beta: float = 0.14,
        n_samples: int = 2500,
        opponent_weight_start: float = 0.06,
        opponent_weight_end: float = 0.42,
        endgame_history_start: float = 0.95,
        opp_min_slack: float = 0.08,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.n_samples = n_samples
        self.opponent_weight_start = opponent_weight_start
        self.opponent_weight_end = opponent_weight_end
        self.endgame_history_start = endgame_history_start
        self.opp_min_slack = opp_min_slack

        self._rng = random.Random()

        # Outcome / utility caches
        self._outcomes: list[Any] = []
        self._my_u: dict[Any, float] = {}
        self._outcome_values: dict[Any, list[Any]] = {}

        # Preferences
        self._reserved_value = 0.0
        self._best_seen = 0.0
        self._max_utility = 1.0
        self._best_outcome = None

        # Domain metadata
        self._issue_names: list[str] = []
        self._issue_importance: list[float] = []

        # Opponent model
        self._received_offers = 0
        self._value_counts: list[defaultdict[Any, int]] = []
        self._issue_stability_counts: list[int] = []
        self._last_received_values: list[Any] | None = None
        self._opponent_first_offer = None
        self._opponent_first_values: list[Any] | None = None
        self._opponent_history: list[Any] = []
        self._opponent_history_set: set[Any] = set()
        self._opponent_best_concession = 0.0

    # -----------------------------
    # Initialization / preprocessing
    # -----------------------------
    def on_preferences_changed(self, changes=None):
        if self.ufun is None or self.nmi is None:
            return

        self._reserved_value = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        self._best_seen = self._reserved_value

        self._outcomes = []
        self._my_u = {}
        self._outcome_values = {}
        self._best_outcome = None

        self._issue_names = []
        self._issue_importance = []

        self._received_offers = 0
        self._value_counts = []
        self._issue_stability_counts = []
        self._last_received_values = None
        self._opponent_first_offer = None
        self._opponent_first_values = None
        self._opponent_history = []
        self._opponent_history_set = set()
        self._opponent_best_concession = 0.0

        try:
            issues = list(self.nmi.outcome_space.issues)
            self._issue_names = [getattr(i, "name", str(i)) for i in issues]
            self._value_counts = [defaultdict(int) for _ in issues]
            self._issue_stability_counts = [0 for _ in issues]
        except Exception:
            pass

        try:
            cardinality = getattr(self.nmi.outcome_space, "cardinality", None)
            if cardinality is not None and cardinality > self.n_samples:
                outcomes = list(self.nmi.outcome_space.sample(self.n_samples))
            else:
                outcomes = list(self.nmi.outcome_space.enumerate())
        except Exception:
            outcomes = []

        feasible = []
        for o in outcomes:
            try:
                u = float(self.ufun(o))
            except Exception:
                continue

            self._my_u[o] = u
            vals = self._outcome_to_values(o)
            if vals is not None:
                self._outcome_values[o] = vals

            if u >= self._reserved_value:
                feasible.append(o)

        feasible.sort(key=lambda o: self._my_u[o], reverse=True)
        self._outcomes = feasible

        if self._outcomes:
            self._best_outcome = self._outcomes[0]
            self._max_utility = self._my_u[self._best_outcome]
        else:
            self._best_outcome = None
            self._max_utility = max(self._reserved_value, 1.0)

        self._estimate_issue_importance()

    # -----------------------------
    # Outcome helpers
    # -----------------------------
    def _outcome_to_values(self, outcome):
        if outcome is None:
            return None

        if isinstance(outcome, dict):
            if self._issue_names:
                return [outcome.get(name, None) for name in self._issue_names]
            try:
                issues = list(self.nmi.outcome_space.issues)
                return [outcome.get(getattr(issue, "name", issue), outcome.get(issue, None)) for issue in issues]
            except Exception:
                return list(outcome.values())

        if isinstance(outcome, (tuple, list)):
            return list(outcome)

        return None

    def _distance(self, a, b) -> int:
        va = self._outcome_values.get(a) or self._outcome_to_values(a)
        vb = self._outcome_values.get(b) or self._outcome_to_values(b)
        if va is None or vb is None:
            return 0
        return sum(1 for x, y in zip(va, vb) if x != y)

    def _normalize_my_utility(self, u: float) -> float:
        denom = max(1e-9, self._max_utility - self._reserved_value)
        return max(0.0, min(1.0, (u - self._reserved_value) / denom))

    # -----------------------------
    # Issue importance
    # -----------------------------
    def _estimate_issue_importance(self):
        """
        Estimate issue importance from own utility landscape.

        For each issue, group feasible outcomes by that issue's value.
        Compute the mean utility for each value, then take the standard deviation
        of those means. Higher deviation => issue value matters more.
        """
        if not self._outcomes:
            self._issue_importance = []
            return

        n_issues = 0
        for o in self._outcomes:
            vals = self._outcome_values.get(o)
            if vals is not None:
                n_issues = len(vals)
                break

        if n_issues <= 0:
            self._issue_importance = []
            return

        scores = []
        for i in range(n_issues):
            groups = defaultdict(list)
            for o in self._outcomes:
                vals = self._outcome_values.get(o)
                if vals is None or i >= len(vals):
                    continue
                groups[vals[i]].append(self._my_u[o])

            means = [sum(vs) / len(vs) for vs in groups.values() if vs]
            if len(means) <= 1:
                scores.append(1.0)
            else:
                scores.append(float(pstdev(means)) + 1e-9)

        s = sum(scores)
        if s <= 0:
            self._issue_importance = [1.0 / len(scores) for _ in scores]
        else:
            self._issue_importance = [x / s for x in scores]

    def _issue_order_least_to_most_important(self) -> list[int]:
        if not self._issue_importance:
            return []
        return sorted(range(len(self._issue_importance)), key=lambda i: self._issue_importance[i])

    # -----------------------------
    # Opponent model
    # -----------------------------
    def _update_opponent_model(self, offer):
        values = self._outcome_values.get(offer)
        if values is None:
            values = self._outcome_to_values(offer)
        if values is None:
            return

        if self._opponent_first_offer is None:
            self._opponent_first_offer = offer
            self._opponent_first_values = list(values)

        if offer not in self._opponent_history_set:
            self._opponent_history.append(offer)
            self._opponent_history_set.add(offer)

        self._received_offers += 1

        if self._value_counts:
            for i, v in enumerate(values[: len(self._value_counts)]):
                self._value_counts[i][v] += 1

        if self._last_received_values is not None and self._issue_stability_counts:
            for i, (old, new) in enumerate(zip(self._last_received_values, values)):
                if old == new:
                    self._issue_stability_counts[i] += 1

        self._last_received_values = list(values)

        if self._opponent_first_offer is not None:
            dist = self._distance(offer, self._opponent_first_offer)
            n = max(1, len(values))
            concession = dist / n
            if concession > self._opponent_best_concession:
                self._opponent_best_concession = concession

    def _estimated_opponent_issue_weights(self) -> list[float]:
        """
        Two signals:
        1. value frequency per issue
        2. issue stability across consecutive offers
        """
        if not self._issue_stability_counts:
            if self._value_counts:
                n = len(self._value_counts)
                return [1.0 / n for _ in range(n)]
            return []

        raw = [x + 1 for x in self._issue_stability_counts]
        s = sum(raw)
        if s <= 0:
            return [1.0 / len(raw) for _ in raw]
        return [x / s for x in raw]

    def _opponent_score(self, outcome) -> float:
        values = self._outcome_values.get(outcome)
        if values is None:
            values = self._outcome_to_values(outcome)

        if values is None or not self._value_counts or self._received_offers == 0:
            return 0.0

        weights = self._estimated_opponent_issue_weights()
        if not weights:
            weights = [1.0 / len(values) for _ in values]

        score = 0.0
        m = min(len(values), len(self._value_counts))
        for i in range(m):
            counts = self._value_counts[i]
            total = sum(counts.values())
            if total <= 0:
                continue
            score += weights[i] * (counts[values[i]] / total)

        # Extra anchor toward opponent's first offer
        if self._opponent_first_offer is not None:
            dist = self._distance(outcome, self._opponent_first_offer)
            anchor = 1.0 - dist / max(1, m)
            score = 0.75 * score + 0.25 * anchor

        return float(max(0.0, min(1.0, score)))

    def _opponent_concession_level(self) -> float:
        return float(max(0.0, min(1.0, self._opponent_best_concession)))

    # -----------------------------
    # Concession / thresholds
    # -----------------------------
    def _target(self, t: float) -> float:
        if not self._outcomes:
            return self._reserved_value

        span = self._max_utility - self._reserved_value
        if span <= 0:
            return self._reserved_value

        # Base Boulware curve
        target = self._max_utility - span * (t ** (1.0 / self.beta))

        # Reciprocate slightly if the opponent has clearly conceded
        opp_conc = self._opponent_concession_level()
        target -= 0.04 * span * opp_conc * max(0.0, t - 0.35)

        # Endgame softening
        if t > 0.88:
            target -= 0.03 * span
        if t > 0.95:
            target -= 0.04 * span
        if t > 0.985:
            target -= 0.05 * span

        return max(self._reserved_value, target)

    def _acceptance_threshold(self, t: float) -> float:
        target = self._target(t)
        span = max(0.0, self._max_utility - self._reserved_value)

        if t > 0.995:
            return self._reserved_value
        if t > 0.98:
            return max(self._reserved_value, target - 0.08 * span)
        if t > 0.94:
            return max(self._reserved_value, target - 0.04 * span)
        return target

    def _opponent_weight(self, t: float) -> float:
        base = self.opponent_weight_start + (
            self.opponent_weight_end - self.opponent_weight_start
        ) * t

        # Increase opponent orientation if they actually concede
        base += 0.10 * self._opponent_concession_level() * max(0.0, t - 0.5)
        return float(max(0.0, min(0.7, base)))

    def _min_acceptable_opponent_score(self, t: float) -> float:
        """
        Lower bound on estimated opponent acceptability for offers we send.
        Relax this gradually over time.
        """
        return max(0.10, 0.48 - 0.35 * t - self.opp_min_slack * self._opponent_concession_level())

    # -----------------------------
    # Structured bid construction
    # -----------------------------
    def _candidate_pool(self, target: float) -> list[Any]:
        if not self._outcomes:
            return []
        return [o for o in self._outcomes if self._my_u[o] >= target]

    def _issue_value_choices(self, issue_idx: int, target: float, limit: int = 6) -> list[Any]:
        """
        Return a small set of promising values for a given issue:
        values that appear in feasible outcomes, ordered by mean self utility.
        """
        groups = defaultdict(list)
        for o in self._outcomes:
            u = self._my_u[o]
            if u < target:
                continue
            vals = self._outcome_values.get(o)
            if vals is None or issue_idx >= len(vals):
                continue
            groups[vals[issue_idx]].append(o)

        scored = []
        for v, os in groups.items():
            mu = sum(self._my_u[o] for o in os) / len(os)
            opp = sum(self._opponent_score(o) for o in os) / len(os)
            scored.append((0.8 * mu + 0.2 * opp, v))

        scored.sort(reverse=True)
        return [v for _, v in scored[:limit]]

    def _modify_reference_bid(self, reference, target: float, t: float):
        """
        AhBuNe-style structural bidding:
        start from a strong reference bid and selectively modify issues.

        Early: change few least-important issues
        Late: change more issues, more opponent-oriented values
        """
        if reference is None:
            return None

        ref_vals = self._outcome_values.get(reference)
        if ref_vals is None:
            ref_vals = self._outcome_to_values(reference)
        if ref_vals is None:
            return None

        least_to_most = self._issue_order_least_to_most_important()
        if not least_to_most:
            return reference

        n_issues = len(ref_vals)
        # Number of issues to try changing increases with time
        k = max(1, min(n_issues, int(math.ceil(t * n_issues))))
        chosen_issue_idxs = least_to_most[:k]

        feasible = self._candidate_pool(target)
        if not feasible:
            return reference

        candidate_set = set(feasible)
        best = None
        best_score = -math.inf
        ow = self._opponent_weight(t)
        opp_floor = self._min_acceptable_opponent_score(t)

        # Build candidate bids by replacing selected issues with good values
        # drawn from feasible outcomes / opponent first offer when available.
        tries = 0
        max_tries = 60

        while tries < max_tries:
            tries += 1
            vals = list(ref_vals)

            for idx in chosen_issue_idxs:
                options = self._issue_value_choices(idx, target, limit=6)
                if not options:
                    continue

                # Prefer opponent anchor values more later in the negotiation
                if (
                    self._opponent_first_values is not None
                    and idx < len(self._opponent_first_values)
                    and self._rng.random() < (0.15 + 0.45 * t)
                ):
                    anchor_val = self._opponent_first_values[idx]
                    if anchor_val in options:
                        vals[idx] = anchor_val
                        continue

                vals[idx] = self._rng.choice(options)

            proposal = tuple(vals) if isinstance(reference, tuple) else vals

            # Convert list back to dict if needed
            if isinstance(reference, dict):
                proposal = {
                    self._issue_names[i] if i < len(self._issue_names) else i: vals[i]
                    for i in range(len(vals))
                }

            # Only keep candidates that exist in our evaluated set if possible
            if proposal not in candidate_set:
                # Try to map structurally similar feasible outcomes instead
                nearby = []
                for o in feasible[: min(250, len(feasible))]:
                    d = self._distance(o, proposal)
                    nearby.append((d, o))
                if not nearby:
                    continue
                nearby.sort(key=lambda x: x[0])
                proposal = nearby[0][1]

            my_u = self._my_u.get(proposal)
            if my_u is None or my_u < target:
                continue

            opp_u = self._opponent_score(proposal)
            if opp_u < opp_floor and t < 0.97:
                continue

            my_n = self._normalize_my_utility(my_u)
            score = (1.0 - ow) * my_n + ow * opp_u

            # Small bonus for matching opponent anchor later
            if self._opponent_first_offer is not None:
                dist = self._distance(proposal, self._opponent_first_offer)
                score += 0.05 * t * (1.0 - dist / max(1, n_issues))

            # Tiny noise to prevent identical paths every run
            score += self._rng.uniform(0.0, 1e-8)

            if score > best_score:
                best_score = score
                best = proposal

        return best if best is not None else reference

    def _best_from_opponent_history(self, t: float):
        """
        Endgame reuse of opponent offers:
        pick the best opponent-offered bid that is good for us and plausibly acceptable to them.
        """
        if not self._opponent_history:
            return None

        target = self._target(t)
        feasible_hist = []
        for o in self._opponent_history:
            try:
                u = self._my_u.get(o, float(self.ufun(o)))
            except Exception:
                continue
            if u >= self._reserved_value:
                feasible_hist.append(o)

        if not feasible_hist:
            return None

        ow = self._opponent_weight(t)
        best = None
        best_score = -math.inf

        for o in feasible_hist:
            my_u = self._my_u.get(o, float(self.ufun(o)))
            opp_u = self._opponent_score(o)
            my_n = self._normalize_my_utility(my_u)

            # Stronger preference for already-seen opponent bids near deadline
            score = (1.0 - ow) * my_n + ow * opp_u
            if my_u >= target:
                score += 0.08
            score += 0.05 * t

            if score > best_score:
                best_score = score
                best = o

        return best

    def _choose_offer(self, target: float, t: float):
        if not self._outcomes:
            return None

        # Endgame: aggressively try best opponent-history bids
        if t >= self.endgame_history_start:
            hist = self._best_from_opponent_history(t)
            if hist is not None and float(self.ufun(hist)) >= self._reserved_value:
                return hist

        # Reference bid: best own bid early, increasingly opponent-friendly later
        reference = self._best_outcome
        if reference is None:
            return self._outcomes[0]

        if t > 0.60:
            pool = self._candidate_pool(max(target, self._reserved_value))
            if pool:
                ow = self._opponent_weight(t)
                best = None
                best_score = -math.inf
                for o in pool[: min(80, len(pool))]:
                    my_n = self._normalize_my_utility(self._my_u[o])
                    opp_u = self._opponent_score(o)
                    score = (1.0 - ow) * my_n + ow * opp_u
                    if score > best_score:
                        best_score = score
                        best = o
                if best is not None:
                    reference = best

        offer = self._modify_reference_bid(reference, target, t)
        if offer is not None and float(self.ufun(offer)) >= self._reserved_value:
            return offer

        # Fallback: frontier scoring
        candidates = self._candidate_pool(target)
        if not candidates:
            return self._outcomes[-1]

        frontier = candidates[: min(60, len(candidates))]
        ow = self._opponent_weight(t)
        opp_floor = self._min_acceptable_opponent_score(t)

        best = None
        best_score = -math.inf
        for o in frontier:
            my_n = self._normalize_my_utility(self._my_u[o])
            opp_u = self._opponent_score(o)

            if opp_u < opp_floor and t < 0.97:
                continue

            score = (1.0 - ow) * my_n + ow * opp_u
            score += self._rng.uniform(0.0, 1e-8)

            if score > best_score:
                best_score = score
                best = o

        return best if best is not None else frontier[0]

    # -----------------------------
    # Acceptance
    # -----------------------------
    def _should_accept(self, offer, t: float) -> bool:
        if offer is None or self.ufun is None:
            return False

        try:
            offer_u = self._my_u.get(offer, float(self.ufun(offer)))
        except Exception:
            return False

        if offer_u < self._reserved_value:
            return False

        if offer_u > self._best_seen:
            self._best_seen = offer_u

        # 1. Standard threshold acceptance
        if offer_u >= self._acceptance_threshold(t):
            return True

        # 2. Accept if current offer dominates what we would send next
        next_offer = self._choose_offer(self._target(t), t)
        if next_offer is not None:
            next_u = self._my_u.get(next_offer, float(self.ufun(next_offer)))
            if offer_u >= next_u:
                return True
            if t > 0.92 and offer_u >= next_u - 0.02:
                return True

        # 3. Endgame: if this is one of the best opponent-history offers, take it
        if t >= self.endgame_history_start:
            hist_best = self._best_from_opponent_history(t)
            if hist_best is not None:
                hist_u = self._my_u.get(hist_best, float(self.ufun(hist_best)))
                if offer_u >= hist_u - 1e-9:
                    return True

        # 4. Regret minimization
        if t > 0.96 and offer_u >= self._best_seen:
            return True

        # 5. Final safety valve
        if t > 0.985 and offer_u >= self._reserved_value:
            return True

        return False

    # -----------------------------
    # Main step
    # -----------------------------
    def __call__(self, state: SAOState) -> SAOResponse:
        t = float(state.relative_time)

        if state.current_offer is not None:
            self._update_opponent_model(state.current_offer)

            if self._should_accept(state.current_offer, t):
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        target = self._target(t)
        offer = self._choose_offer(target, t)

        if offer is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        return SAOResponse(ResponseType.REJECT_OFFER, offer)