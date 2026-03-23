from pathlib import Path
from negmas import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.inout import Scenario
from negmas.preferences import LinearAdditiveUtilityFunction as UFun
from negmas.preferences.value_fun import IdentityFun, AffineFun
from negmas.sao.negotiators import BoulwareTBNegotiator as Boulware
from negmas.sao.negotiators import LinearTBNegotiator as Linear
from negmas.sao.negotiators import NaiveTitForTatNegotiator as Naive
from negmas.sao.negotiators import HybridNegotiator as Hybrid
from negmas.sao.negotiators import MiCRONegotiator as micro
from negmas.tournaments.neg import cartesian_tournament
from Group34_Negotiator_boulware import Group34_Negotiator  # top-level import
from Group34_Negotiator_hybrid import Group34_Negotiator_Hybrid

def my_agent_factory(ufun):
    return Group34_Negotiator(ufun=ufun)

if __name__ == "__main__":
    path = Path.home()
    issues = [make_issue(name="price", values=(0, 50))]
    os = make_os(issues=issues)

    seller_ufun = UFun(values=[IdentityFun()], outcome_space=os).normalize()
    buyer_ufun = UFun(values=[AffineFun(slope=-1, bias=50)], outcome_space=os).normalize()
    scenario = Scenario(outcome_space=os, ufuns=[seller_ufun, buyer_ufun])

    competitors = [Boulware, Linear, Naive, Hybrid, Group34_Negotiator_Hybrid]
    competitors2 = [Boulware, Group34_Negotiator]

    results = cartesian_tournament(
        competitors=competitors,
        scenarios=[scenario],
        n_repetitions=1,
        n_steps=1000,
        path=path,
        time_limit=1
    )

    print(results.scores)