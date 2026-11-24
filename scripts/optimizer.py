"""
IPL Squad Optimization Engine
Week 3 — Constrained Squad Selection via Integer Linear Programming

Uses PuLP with HiGHS solver to select optimal squads under real IPL auction rules:
  - Budget cap (default ₹120 Cr)
  - Squad size limits (18-25)
  - Max 8 overseas players
  - Role minimums: 4 batsmen, 4 bowlers, 2 all-rounders, 2 wicketkeepers

Supports three objective functions:
  1. composite_value — balanced performance + value score
  2. performance_score — pure on-field performance
  3. value_score — bargain hunting (predicted_price / sold_price)
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, lpSum, getSolver

SOLVER = getSolver('HiGHS', msg=False)

IPL_RULES = {
    'budget_cr': 120.0,
    'max_squad_size': 25,
    'min_squad_size': 18,
    'max_overseas': 8,
    'min_batsmen': 4,
    'min_bowlers': 4,
    'min_allrounders': 2,
    'min_wicketkeepers': 2,
}


def compute_performance_score(row):
    """
    Composite score (0-100) weighting batting and bowling by player role.

    Batting components: career runs, strike rate, recent form, sixes, awards
    Bowling components: career wickets, economy, recent form, awards
    """
    bat_score = 0
    bowl_score = 0

    # Batting sub-score
    if row.get('career_runs', 0) > 0:
        runs_norm = min(row['career_runs'] / 5000, 1.0) * 25
        sr_norm = min(max(row.get('career_batting_sr', 100) - 100, 0) / 60, 1.0) * 25
        recent_norm = min(row.get('recent_runs', 0) / 800, 1.0) * 30
        six_norm = min(row.get('career_sixes', 0) / 100, 1.0) * 10
        potm_bat = min(row.get('total_potm', 0) / 10, 1.0) * 10
        bat_score = runs_norm + sr_norm + recent_norm + six_norm + potm_bat

    # Bowling sub-score
    if row.get('career_wickets', 0) > 0:
        wickets_norm = min(row['career_wickets'] / 150, 1.0) * 30
        econ = row.get('career_economy', 8)
        econ_norm = max(1 - (econ - 6) / 4, 0) * 25 if econ > 0 else 0.5 * 25
        recent_bowl = min(row.get('recent_wickets', 0) / 30, 1.0) * 30
        potm_bowl = min(row.get('total_potm', 0) / 10, 1.0) * 15
        bowl_score = wickets_norm + econ_norm + recent_bowl + potm_bowl

    # Weight by role
    role = row.get('role', 'Batsman')
    if role == 'All-Rounder':
        return bat_score * 0.5 + bowl_score * 0.5
    elif role == 'Bowler':
        return bowl_score * 0.8 + bat_score * 0.2
    elif role == 'Wicketkeeper':
        return bat_score * 0.9 + bowl_score * 0.1
    else:  # Batsman
        return bat_score * 0.9 + bowl_score * 0.1


def compute_value_score(row):
    """Value = predicted fair price / actual sold price. >1 means underpaid."""
    if row.get('sold_price_cr', 0) > 0 and row.get('predicted_price_cr', 0) > 0:
        return row['predicted_price_cr'] / row['sold_price_cr']
    return 1.0


def prepare_player_pool(predictions_df, auction_year=2025):
    """Filter to target auction year and compute scoring columns."""
    pool = predictions_df[predictions_df['auction_year'] == auction_year].copy()
    pool['performance_score'] = pool.apply(compute_performance_score, axis=1)
    pool['value_score'] = pool.apply(compute_value_score, axis=1).clip(upper=5)
    pool['composite_value'] = (
        pool['performance_score'] * 0.6 +
        pool['value_score'] * 20 * 0.4
    )
    return pool.reset_index(drop=True)


def optimise_squad(player_pool, budget=120.0, rules=None,
                   objective='composite_value', retained_players=None):
    """
    Integer linear programming optimiser for IPL squad selection.

    Args:
        player_pool: DataFrame with player stats and scoring columns
        budget: Total budget in crores
        rules: Dict of constraints (uses IPL_RULES if None)
        objective: Column to maximise ('composite_value', 'performance_score', 'value_score')
        retained_players: List of player names to auto-include

    Returns:
        Dict with squad DataFrame, budget details, and constraint satisfaction
    """
    if rules is None:
        rules = IPL_RULES.copy()

    pool = player_pool.copy().reset_index(drop=True)

    # Handle retained players
    retained = []
    retained_cost = 0
    retained_overseas = 0
    if retained_players:
        for name in retained_players:
            match = pool[pool['player'] == name]
            if len(match) > 0:
                row = match.iloc[0]
                retained.append(row.to_dict())
                retained_cost += row['sold_price_cr']
                if row.get('player_origin') == 'Overseas':
                    retained_overseas += 1
        pool = pool[~pool['player'].isin(retained_players)].reset_index(drop=True)

    n = len(pool)
    remaining_budget = budget - retained_cost
    remaining_slots = rules['max_squad_size'] - len(retained)
    remaining_overseas = rules['max_overseas'] - retained_overseas

    # Build ILP model
    prob = LpProblem("IPL_Squad_Optimiser", LpMaximize)
    x = [prob.add_variable(f"x_{i}", cat='Binary') for i in range(n)]

    # Objective: maximise total score
    prob += lpSum(x[i] * float(pool.iloc[i][objective]) for i in range(n))

    # Budget constraint
    prob += lpSum(x[i] * float(pool.iloc[i]['sold_price_cr']) for i in range(n)) <= remaining_budget

    # Squad size constraints
    prob += lpSum(x[i] for i in range(n)) <= remaining_slots
    prob += lpSum(x[i] for i in range(n)) >= max(rules['min_squad_size'] - len(retained), 0)

    # Overseas limit
    overseas_idx = [i for i in range(n) if pool.iloc[i].get('player_origin') == 'Overseas']
    if overseas_idx:
        prob += lpSum(x[i] for i in overseas_idx) <= remaining_overseas

    # Role minimums
    retained_roles = {}
    for r in retained:
        role = r.get('role', '')
        retained_roles[role] = retained_roles.get(role, 0) + 1

    role_constraints = [
        ('Batsman', rules['min_batsmen']),
        ('Bowler', rules['min_bowlers']),
        ('All-Rounder', rules['min_allrounders']),
        ('Wicketkeeper', rules['min_wicketkeepers']),
    ]
    for role, minimum in role_constraints:
        need = max(minimum - retained_roles.get(role, 0), 0)
        role_idx = [i for i in range(n) if pool.iloc[i].get('role') == role]
        if role_idx and need > 0:
            prob += lpSum(x[i] for i in role_idx) >= need

    # Solve
    result = prob.solve(SOLVER)

    # Extract solution
    selected = [pool.iloc[i].to_dict() for i in range(n) if x[i].value() == 1]
    squad = retained + selected
    total_cost = sum(p['sold_price_cr'] for p in squad)

    squad_df = pd.DataFrame(squad)
    overseas_count = len(squad_df[squad_df['player_origin'] == 'Overseas']) if len(squad_df) > 0 else 0

    return {
        'status': 'optimal' if len(selected) > 0 else 'infeasible',
        'squad': squad_df,
        'total_players': len(squad),
        'budget_used': round(total_cost, 2),
        'budget_remaining': round(budget - total_cost, 2),
        'total_score': round(sum(p.get(objective, 0) for p in squad), 2),
        'overseas_count': overseas_count,
        'retained_count': len(retained),
    }


def format_squad(result, title="OPTIMISED SQUAD"):
    """Pretty-print squad selection results."""
    sq = result['squad']
    lines = []
    budget_total = result['budget_used'] + result['budget_remaining']

    lines.append(f"\n{'='*78}")
    lines.append(f"  {title}")
    lines.append(f"{'='*78}")
    lines.append(f"  Players: {result['total_players']} | "
                 f"Budget: ₹{result['budget_used']:.1f}/₹{budget_total:.0f} Cr | "
                 f"Overseas: {result['overseas_count']}/{IPL_RULES['max_overseas']}")
    if result['retained_count'] > 0:
        lines.append(f"  Retained: {result['retained_count']}")

    lines.append(f"\n  {'Player':<28} {'Role':<14} {'Origin':<10} {'Price':>7} {'Perf':>6} {'Value':>6}")
    lines.append(f"  {'-'*74}")

    for _, p in sq.sort_values('sold_price_cr', ascending=False).iterrows():
        origin = str(p.get('player_origin', '?'))
        lines.append(
            f"  {p['player']:<28} {str(p.get('role','?')):<14} {origin:<10} "
            f"₹{p['sold_price_cr']:>5.1f}  {p.get('performance_score',0):>5.1f}  "
            f"{p.get('value_score',0):>5.2f}"
        )

    lines.append(f"\n  Role Breakdown:")
    for role in ['Batsman', 'Bowler', 'All-Rounder', 'Wicketkeeper']:
        rp = sq[sq['role'] == role]
        if len(rp) > 0:
            names = ', '.join(rp.sort_values('sold_price_cr', ascending=False)['player'].tolist())
            lines.append(f"    {role} ({len(rp)}): ₹{rp['sold_price_cr'].sum():.1f} Cr")
            lines.append(f"      {names}")

    return '\n'.join(lines)


if __name__ == '__main__':
    import os

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predictions = pd.read_csv(os.path.join(PROJECT_DIR, 'data/processed/predictions.csv'))

    pool = prepare_player_pool(predictions, auction_year=2025)
    print(f"IPL 2025 Auction Pool: {len(pool)} players")
    print(f"  Indian: {(pool['player_origin']=='Indian').sum()}, "
          f"Overseas: {(pool['player_origin']=='Overseas').sum()}")

    # Scenario 1: Full budget, best composite squad
    r1 = optimise_squad(pool, budget=120.0, objective='composite_value')
    print(format_squad(r1, "SCENARIO 1: Best Overall Squad (₹120 Cr)"))

    # Scenario 2: Value hunting on ₹80 Cr budget
    r2 = optimise_squad(pool, budget=80.0, objective='composite_value')
    print(format_squad(r2, "SCENARIO 2: Value Squad (₹80 Cr Budget)"))

    # Scenario 3: Retain stars and build around them
    r3 = optimise_squad(pool, budget=120.0, objective='composite_value',
                        retained_players=['Virat Kohli', 'Jasprit Bumrah'])
    print(format_squad(r3, "SCENARIO 3: Retain Kohli + Bumrah + Build"))

    # Scenario 4: Pure performance maximisation
    r4 = optimise_squad(pool, budget=120.0, objective='performance_score')
    print(format_squad(r4, "SCENARIO 4: Performance-First Squad (₹120 Cr)"))

    print(f"\n{'='*78}")
    print("  OPTIMIZER SUMMARY")
    print(f"{'='*78}")
    for name, res in [("Best Overall", r1), ("Value (₹80Cr)", r2),
                       ("Kohli+Bumrah", r3), ("Performance", r4)]:
        print(f"  {name:<20} | {res['total_players']} players | "
              f"₹{res['budget_used']:.1f} Cr | {res['overseas_count']} overseas | "
              f"Score: {res['total_score']:.1f}")

    print("\n✅ Optimization complete")
