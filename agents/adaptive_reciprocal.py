import random
from collections import defaultdict

from negmas.sao import SAONegotiator, SAOResponse, ResponseType


class AdaptiveReciprocalNegotiator(SAONegotiator):
    def __init__(self, *args, beta=0.2, n_samples=2500, **kwargs):
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
        self._counts = []
        self._stability = []
        self._last_vals = None
        self._received = 0
        self._best_seen_u = 0.0
        self._last_offer_u = None
        self._credit = 0.0

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
        self._max_u = self._my_u[self._outcomes[0]] if self._outcomes else max(1.0, self._res_val)
        self._best_seen_u = self._res_val

        issues = list(space.issues)
        self._counts = [defaultdict(int) for _ in issues]
        self._stability = [0 for _ in issues]

    def _get_vals(self, outcome):
        if isinstance(outcome, dict):
            return list(outcome.values())
        return list(outcome) if outcome is not None else None

    def _update_opponent(self, offer):
        vals = self._get_vals(offer)
        if not vals:
            return

        self._received += 1
        for i, value in enumerate(vals):
            if i < len(self._counts):
                self._counts[i][value] += 1
                if self._last_vals is not None and i < len(self._last_vals) and value == self._last_vals[i]:
                    self._stability[i] += 1
        self._last_vals = vals

    def _opp_score(self, outcome):
        vals = self._get_vals(outcome)
        if not vals or self._received == 0:
            return 0.0

        m = min(len(vals), len(self._counts))
        if m == 0:
            return 0.0

        total_stab = sum(self._stability[:m]) + m
        weights = [(self._stability[i] + 1.0) / total_stab for i in range(m)]

        score = 0.0
        for i in range(m):
            counts = self._counts[i]
            total = sum(counts.values())
            score += weights[i] * ((counts[vals[i]] + 1.0) / (total + len(counts) + 1.0))
        return score

    def _update_credit(self, u_offer):
        if self._last_offer_u is not None:
            delta = u_offer - self._last_offer_u
            if delta > 0:
                self._credit = 0.7 * self._credit + delta
            else:
                self._credit *= 0.85
        self._last_offer_u = u_offer

    def _target(self, t):
        span = self._max_u - self._res_val
        target = self._max_u - span * (t ** (1.0 / self.beta))
        if t > 0.2:
            target -= min(0.08 * span, 0.45 * self._credit)
        if t > 0.92:
            target -= 0.04 * span
        return max(self._res_val, target)

    def _choose_offer(self, target, t):
        if not self._outcomes:
            return None

        candidates = [o for o in self._outcomes if self._my_u[o] >= target]
        if not candidates:
            candidates = self._outcomes[-1:]

        span = max(1e-9, self._max_u - self._res_val)
        opp_weight = 0.15 + 0.3 * t

        def score(outcome):
            own = (self._my_u[outcome] - self._res_val) / span
            return (
                (1.0 - opp_weight) * own
                + opp_weight * self._opp_score(outcome)
                + self._rng.uniform(0.0, 1e-8)
            )

        return max(candidates[:100], key=score)

    def __call__(self, state):
        t = float(state.relative_time)
        offer = state.current_offer

        if offer is not None:
            self._update_opponent(offer)
            u_offer = self._my_u.get(offer, float(self.ufun(offer)))
            self._update_credit(u_offer)
            self._best_seen_u = max(self._best_seen_u, u_offer)

            target = self._target(t)
            next_bid = self._choose_offer(target, t)
            next_u = self._my_u.get(next_bid, self._res_val) if next_bid is not None else self._res_val

            if u_offer >= target:
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            if t > 0.9 and u_offer >= next_u:
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            if t > 0.97 and u_offer >= self._res_val:
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self._choose_offer(self._target(t), t)
        if bid is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)
