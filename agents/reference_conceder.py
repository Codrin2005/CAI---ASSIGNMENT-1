from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import pstdev

from negmas.sao import SAONegotiator, SAOResponse, ResponseType


class ReferenceConcederNegotiator(SAONegotiator):
    """
    AhBuNe-inspired benchmark:
    structured concession from a strong reference outcome.
    """

    def __init__(self, *args, beta=0.18, n_samples=2500, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.n_samples = n_samples
        self._rng = random.Random()
        self._reset()

    def _reset(self):
        self._outcomes = []
        self._my_u = {}
        self._res_val = 0.0
        self._max_u = 1.0
        self._best_outcome = None
        self._issue_importance = []
        self._opp_first_offer = None
        self._opp_first_vals = None

    def on_preferences_changed(self, changes=None):
        self._reset()
        if not self.ufun or not self.nmi:
            return

        space = self.nmi.outcome_space
        card = getattr(space, "cardinality", 0)
        raw = list(space.sample(self.n_samples) if card > self.n_samples else space.enumerate())

        self._res_val = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        for outcome in raw:
            utility = float(self.ufun(outcome))
            if utility >= self._res_val:
                self._my_u[outcome] = utility
                self._outcomes.append(outcome)

        self._outcomes.sort(key=lambda o: self._my_u[o], reverse=True)
        if self._outcomes:
            self._best_outcome = self._outcomes[0]
            self._max_u = self._my_u[self._best_outcome]
        self._issue_importance = self._estimate_issue_importance()

    def _get_vals(self, outcome):
        if isinstance(outcome, dict):
            return list(outcome.values())
        return list(outcome) if outcome is not None else None

    def _estimate_issue_importance(self):
        if not self._outcomes:
            return []

        ref_vals = self._get_vals(self._outcomes[0])
        if not ref_vals:
            return []

        scores = []
        for i in range(len(ref_vals)):
            groups = defaultdict(list)
            for outcome in self._outcomes[: min(400, len(self._outcomes))]:
                vals = self._get_vals(outcome)
                if not vals or i >= len(vals):
                    continue
                groups[vals[i]].append(self._my_u[outcome])

            means = [sum(xs) / len(xs) for xs in groups.values() if xs]
            scores.append(float(pstdev(means)) + 1e-9 if len(means) > 1 else 1.0)

        total = sum(scores) + 1e-9
        return [score / total for score in scores]

    def _target(self, t):
        span = self._max_u - self._res_val
        target = self._max_u - span * (t ** (1.0 / self.beta))
        if t > 0.9:
            target -= 0.04 * span
        if t > 0.97:
            target -= 0.03 * span
        return max(self._res_val, target)

    def _structured_candidates(self, target, t):
        if not self._outcomes:
            return []

        candidates = [o for o in self._outcomes if self._my_u[o] >= target]
        if not candidates or self._best_outcome is None or not self._issue_importance:
            return candidates or self._outcomes[-1:]

        ref_vals = self._get_vals(self._best_outcome)
        if not ref_vals:
            return candidates

        issue_order = sorted(range(len(ref_vals)), key=lambda i: self._issue_importance[i])
        allowed = max(1, min(len(ref_vals), int(math.ceil((0.2 + 0.8 * t) * len(ref_vals)))))

        for k in range(allowed, len(ref_vals) + 1):
            modifiable = set(issue_order[:k])
            filtered = []
            for outcome in candidates[: min(200, len(candidates))]:
                vals = self._get_vals(outcome)
                if not vals or len(vals) != len(ref_vals):
                    continue
                if all(i in modifiable or vals[i] == ref_vals[i] for i in range(len(ref_vals))):
                    filtered.append(outcome)
            if filtered:
                return filtered
        return candidates

    def _choose_offer(self, target, t):
        candidates = self._structured_candidates(target, t)
        if not candidates:
            return None

        if self._opp_first_vals is None:
            return candidates[0]

        ref_len = len(self._opp_first_vals)

        def score(outcome):
            vals = self._get_vals(outcome)
            anchor = 0.0
            if vals:
                matches = sum(
                    1 for i in range(min(len(vals), ref_len))
                    if vals[i] == self._opp_first_vals[i]
                )
                anchor = matches / max(1, ref_len)
            own = self._my_u[outcome]
            span = max(1e-9, self._max_u - self._res_val)
            own_n = (own - self._res_val) / span
            return 0.8 * own_n + 0.2 * anchor + self._rng.uniform(0.0, 1e-8)

        return max(candidates[:100], key=score)

    def __call__(self, state):
        t = float(state.relative_time)
        offer = state.current_offer

        if offer is not None:
            vals = self._get_vals(offer)
            if self._opp_first_offer is None:
                self._opp_first_offer = offer
                self._opp_first_vals = vals
            u_offer = self._my_u.get(offer, float(self.ufun(offer)))
            target = self._target(t)
            if u_offer >= target:
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            if t > 0.97 and u_offer >= self._res_val:
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self._choose_offer(self._target(t), t)
        if bid is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)
