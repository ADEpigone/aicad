//! Destroy operators: heuristical way of destroying parts of a full assignment to variables.
//!
//! They are shared by the training of the learned models, which masks the variables it asks the
//! network to predict, and by the local searches, which free them to repair the assignment.

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

use std::collections::HashSet;

use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::modelling::{Problem, VariableIndex};

/// Floor of the scores, to avoid dividing by zero.
const SCORE_EPS: f64 = 1e-8;

/// Trait to implement for a new destroy operator.
pub trait DestroyOperator: Send + Sync {
    /// Returns the index of the variables in `problem` to destroy given the current assignment.
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize>;
}

/// The destroy operators one can choose from, as a plain value that builds the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DestroyKind {
    /// [`RandomDestroy`].
    #[default]
    Random,
    /// [`WorstDestroy`].
    Worst,
    /// [`RelatedDestroy`].
    Related,
    /// [`UnitDestroy`].
    Unit,
}

impl DestroyKind {
    /// Builds the operator of this kind that destroys (on average) `fraction` of the free
    /// variables.
    pub fn build(self, fraction: f64) -> Box<dyn DestroyOperator> {
        match self {
            DestroyKind::Random => Box::new(RandomDestroy { fraction }),
            DestroyKind::Worst => Box::new(WorstDestroy { fraction }),
            DestroyKind::Related => Box::new(RelatedDestroy { fraction }),
            DestroyKind::Unit => Box::new(UnitDestroy { fraction }),
        }
    }
}

/// Uniformly destroys a `fraction` of the free (domain size > 1) variables.
pub struct RandomDestroy {
    /// Fraction of the free variables to destroy.
    pub fraction: f64,
}

impl DestroyOperator for RandomDestroy {
    fn destroy(&self, problem: &Problem, _assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let mut free: Vec<usize> = free_variables(problem);
        if free.is_empty() {
            return free;
        }
        free.shuffle(rng);
        let k = ((free.len() as f64 * self.fraction).round() as usize).clamp(1, free.len());
        free.truncate(k);
        free
    }
}

/// Stochastic worst removal: each free variable is destroyed with a probability that is the
/// sigmoid of its standardised count of currently-violated constraints, rescaled so the mean
/// probability matches `fraction`. Every free variable keeps a non-zero probability.
pub struct WorstDestroy {
    /// Mean fraction of the free variables to destroy.
    pub fraction: f64,
}

impl DestroyOperator for WorstDestroy {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let mut unsat_per_variable = vec![0f64; problem.number_variables()];
        for c in problem.iter_constraints() {
            if !problem[c].is_satisfied(assignment) {
                for v in problem[c].iter_scope() {
                    unsat_per_variable[v.0] += 1.0;
                }
            }
        }
        let free = free_variables(problem);
        if free.is_empty() {
            return free;
        }

        let n = free.len() as f64;
        let values: Vec<f64> = free.iter().map(|&i| unsat_per_variable[i]).collect();
        let mean = values.iter().sum::<f64>() / n;
        let std = if free.len() > 1 {
            (values.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        let probs: Vec<f64> = values
            .iter()
            .map(|s| 1.0 / (1.0 + (-(s - mean) / (std + SCORE_EPS)).exp()))
            .collect();
        let scale = self.fraction / (probs.iter().sum::<f64>() / n + SCORE_EPS);

        free.into_iter()
            .zip(probs)
            .filter(|&(_, p)| rng.random_bool((p * scale).clamp(0.0, 1.0)))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Stochastic related removal (Shaw): draws a random subset of constraints
/// and destroys every variable in their scope. Constraints are drawn
/// independently with a probability rescaled so the expected number of
/// destroyed variables tracks `fraction * n`, regardless of how many
/// constraints the problem has relative to its variables
pub struct RelatedDestroy {
    /// Expected fraction of the free variables to destroy.
    pub fraction: f64,
}

impl DestroyOperator for RelatedDestroy {
    fn destroy(&self, problem: &Problem, _assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let n = problem.number_variables().max(1);
        let m = problem.number_constraints();
        if m == 0 {
            return Vec::new();
        }

        let total_scope: usize = problem
            .iter_constraints()
            .map(|c| problem[c].iter_scope().count())
            .sum();
        let avg_scope = (total_scope as f64 / m as f64).max(1.0);

        // E[# destroyed] ~= p_constraint * m * avg_scope (upper bound, ignores
        // overlap between scopes)
        let p_constraint = (self.fraction * n as f64 / (m as f64 * avg_scope)).clamp(0.0, 1.0);

        let mut destroyed = HashSet::new();
        for c in problem.iter_constraints() {
            if rng.random_bool(p_constraint) {
                for v in problem[c].iter_scope() {
                    if problem[v].domain_size() > 1 {
                        destroyed.insert(v.0);
                    }
                }
            }
        }
        destroyed.into_iter().collect()
    }
}

/// Violation-guided destroy of whole *units*, groups of variables freed together (Shaw, CP 1998;
/// the worst removal of Ropke and Pisinger, Transportation Science 2006).
///
/// The units are the destroy units the modeller declared on the problem
/// ([`Problem::add_destroy_unit`]); without any, the scopes of the constraints that fit in the
/// budget (a constraint over most of the variables would make the repair a restart), an identical
/// scope counted once. In both cases a free variable in no unit is a unit of its own, so that every
/// variable can be destroyed, and the variables with a single value are never destroyed.
///
/// The budget is `fraction` of the free variables (at least one). Each unit is scored with
/// $1 + v_u$, where $v_u$ is the number of constraints violated by `assignment` that have a
/// variable of the unit in their scope, and units are drawn without replacement with
/// probabilities proportional to their scores until the destroyed variables reach the budget:
/// the units of the violated constraints first, but every unit has a chance. Units are destroyed
/// whole, so the last one can overshoot the budget, by less than its own size; one that frees no new
/// variable is skipped.
pub struct UnitDestroy {
    /// Fraction of the free variables to destroy, at least.
    pub fraction: f64,
}

impl UnitDestroy {
    /// The units of `problem` for a budget of `budget` variables, as indices of free variables.
    fn units(problem: &Problem, budget: usize) -> Vec<Vec<usize>> {
        let is_free = |v: usize| problem[VariableIndex(v)].domain_size() > 1;
        let mut units: Vec<Vec<usize>> = if problem.destroy_units().is_empty() {
            let mut scopes: Vec<Vec<usize>> = problem
                .iter_constraints()
                .map(|c| {
                    let mut scope: Vec<usize> = problem[c]
                        .iter_scope()
                        .map(|v| v.0)
                        .filter(|&v| is_free(v))
                        .collect();
                    scope.sort_unstable();
                    scope.dedup();
                    scope
                })
                .filter(|scope| !scope.is_empty() && scope.len() <= budget)
                .collect();
            scopes.sort();
            scopes.dedup();
            scopes
        } else {
            problem
                .destroy_units()
                .iter()
                .map(|unit| unit.iter().map(|v| v.0).filter(|&v| is_free(v)).collect())
                .filter(|unit: &Vec<usize>| !unit.is_empty())
                .collect()
        };
        let mut covered = vec![false; problem.number_variables()];
        for &v in units.iter().flatten() {
            covered[v] = true;
        }
        units.extend(
            free_variables(problem)
                .into_iter()
                .filter(|&v| !covered[v])
                .map(|v| vec![v]),
        );
        units
    }
}

impl DestroyOperator for UnitDestroy {
    fn destroy(&self, problem: &Problem, assignment: &[isize], rng: &mut StdRng) -> Vec<usize> {
        let free = free_variables(problem);
        if free.is_empty() {
            return free;
        }
        let budget = ((free.len() as f64 * self.fraction).round() as usize).clamp(1, free.len());
        let units = Self::units(problem, budget);

        // v_u: the violated constraints with a variable in the unit, each counted once.
        let violated: Vec<bool> = problem
            .iter_constraints()
            .map(|c| !problem[c].is_satisfied(assignment))
            .collect();
        let mut seen = vec![usize::MAX; problem.number_constraints()];
        let scores: Vec<f64> = units
            .iter()
            .enumerate()
            .map(|(u, unit)| {
                let mut count = 0usize;
                for &v in unit {
                    for c in problem[VariableIndex(v)].iter_constraints() {
                        if violated[c.0] && seen[c.0] != u {
                            seen[c.0] = u;
                            count += 1;
                        }
                    }
                }
                1.0 + count as f64
            })
            .collect();

        // Weighted sampling without replacement (Efraimidis and Spirakis): sorting by
        // ln(U) / score, U uniform in (0, 1], draws the units in the order sequential draws
        // proportional to the scores would.
        let mut order: Vec<(f64, usize)> = scores
            .iter()
            .enumerate()
            .map(|(u, &score)| {
                let draw: f64 = 1.0 - rng.random::<f64>();
                (draw.ln() / score, u)
            })
            .collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut destroyed = vec![false; problem.number_variables()];
        let mut result = Vec::with_capacity(budget);
        for (_, u) in order {
            if result.len() >= budget {
                break;
            }
            for &v in &units[u] {
                if !destroyed[v] {
                    destroyed[v] = true;
                    result.push(v);
                }
            }
        }
        result
    }
}

/// Returns the variables that can be destroyed in the problem; that is, the variables with domain
/// size greater than 1.
fn free_variables(problem: &Problem) -> Vec<usize> {
    problem
        .iter_variables()
        .filter(|&v| problem[v].domain_size() > 1)
        .map(|v| v.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::not_equals;
    use rand::SeedableRng;

    fn two_pairs() -> Problem {
        let mut problem = Problem::default();
        let vars: Vec<_> = (0..4)
            .map(|_| problem.add_variable(vec![0, 1, 2]))
            .collect();
        not_equals(&mut problem, vars[0], vars[1]);
        not_equals(&mut problem, vars[2], vars[3]);
        problem
    }

    fn frequencies(
        problem: &Problem,
        assignment: &[isize],
        fraction: f64,
        draws: usize,
    ) -> Vec<f64> {
        let op = WorstDestroy { fraction };
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = vec![0usize; assignment.len()];
        for _ in 0..draws {
            for v in op.destroy(problem, assignment, &mut rng) {
                counts[v] += 1;
            }
        }
        counts
            .into_iter()
            .map(|c| c as f64 / draws as f64)
            .collect()
    }

    #[test]
    fn violated_variables_are_destroyed_more_often_but_not_exclusively() {
        let problem = two_pairs();
        let freq = frequencies(&problem, &[0, 0, 0, 1], 0.5, 20_000);
        assert!((freq[0] - 0.704).abs() < 0.02, "{freq:?}");
        assert!((freq[1] - 0.704).abs() < 0.02, "{freq:?}");
        assert!((freq[2] - 0.296).abs() < 0.02, "{freq:?}");
        assert!((freq[3] - 0.296).abs() < 0.02, "{freq:?}");
    }

    /// Twelve variables over {0, 1, 2}, declared as three units of four, with `x0 != x1` (in the
    /// first unit) and `x8 != x9` (in the third).
    fn three_units() -> Problem {
        let mut problem = Problem::default();
        let vars = problem.add_variables(12, vec![0, 1, 2]);
        not_equals(&mut problem, vars[0], vars[1]);
        not_equals(&mut problem, vars[8], vars[9]);
        for unit in vars.chunks(4) {
            problem.add_destroy_unit(unit.iter().copied());
        }
        problem
    }

    /// Which of the three units of `three_units` a destroyed set is, made of whole units.
    fn units_of(destroyed: &[usize]) -> Vec<usize> {
        let mut units: Vec<usize> = destroyed.iter().map(|v| v / 4).collect();
        units.sort_unstable();
        units.dedup();
        for &u in &units {
            for v in 4 * u..4 * u + 4 {
                assert!(
                    destroyed.contains(&v),
                    "unit {u} destroyed in part: {destroyed:?}"
                );
            }
        }
        assert_eq!(destroyed.len(), 4 * units.len(), "{destroyed:?}");
        units
    }

    #[test]
    fn unit_destroy_frees_whole_units_until_the_budget_and_overshoots_by_less_than_one() {
        let problem = three_units();
        let mut rng = StdRng::seed_from_u64(3);
        let assignment = [0; 12];
        for (fraction, units) in [
            (0.05, 1),
            (0.33, 1),
            (0.38, 2),
            (0.5, 2),
            (0.67, 2),
            (0.75, 3),
        ] {
            let op = UnitDestroy { fraction };
            for _ in 0..50 {
                let destroyed = op.destroy(&problem, &assignment, &mut rng);
                assert_eq!(units_of(&destroyed).len(), units, "fraction {fraction}");
            }
        }
    }

    #[test]
    fn unit_destroy_prefers_the_units_of_violated_constraints() {
        // x0 != x1 is violated, x8 != x9 is not: scores 2, 1, 1, so the first unit is drawn first
        // with probability 1/2, each other one with 1/4.
        let problem = three_units();
        let mut assignment = [0; 12];
        assignment[9] = 1;
        let op = UnitDestroy { fraction: 0.25 };
        let mut rng = StdRng::seed_from_u64(11);
        let mut counts = [0usize; 3];
        let draws = 20_000;
        for _ in 0..draws {
            counts[units_of(&op.destroy(&problem, &assignment, &mut rng))[0]] += 1;
        }
        let freq: Vec<f64> = counts.iter().map(|&c| c as f64 / draws as f64).collect();
        for (f, expected) in freq.iter().zip([0.5, 0.25, 0.25]) {
            assert!((f - expected).abs() < 0.02, "{freq:?}");
        }
    }

    #[test]
    fn without_declared_units_the_scopes_that_fit_the_budget_are_the_units() {
        // An all-different over all six variables is larger than the budget of two and is not a
        // unit; the three not-equals pairs are, so exactly one pair is destroyed.
        let mut problem = Problem::default();
        let vars = problem.add_variables(6, vec![0, 1, 2, 3, 4, 5]);
        crate::constraints::all_different(&mut problem, vars.clone());
        for pair in vars.chunks(2) {
            not_equals(&mut problem, pair[0], pair[1]);
        }
        let op = UnitDestroy { fraction: 0.34 };
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..50 {
            let mut destroyed = op.destroy(&problem, &[0, 1, 2, 3, 4, 5], &mut rng);
            destroyed.sort_unstable();
            assert!(
                [vec![0, 1], vec![2, 3], vec![4, 5]].contains(&destroyed),
                "{destroyed:?}"
            );
        }
    }

    #[test]
    fn variables_in_no_unit_are_units_of_their_own_and_fixed_ones_are_never_destroyed() {
        // Units {x0, x1} and {x1, x2}; x3 is in none, x4 has a single value.
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1]);
        let fixed = problem.add_variable(vec![0]);
        problem.add_destroy_unit([vars[0], vars[1]]);
        problem.add_destroy_unit([vars[1], vars[2], fixed]);
        let op = UnitDestroy { fraction: 1.0 };
        let mut rng = StdRng::seed_from_u64(9);
        let mut destroyed = op.destroy(&problem, &[0; 5], &mut rng);
        destroyed.sort_unstable();
        assert_eq!(destroyed, vec![0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "at least one variable")]
    fn an_empty_destroy_unit_panics() {
        let mut problem = Problem::default();
        problem.add_destroy_unit([]);
    }

    #[test]
    #[should_panic(expected = "only 1 variables")]
    fn a_destroy_unit_over_a_missing_variable_panics() {
        let mut problem = Problem::default();
        problem.add_variable(vec![0, 1]);
        problem.add_destroy_unit([crate::modelling::VariableIndex(3)]);
    }

    #[test]
    fn a_satisfied_assignment_is_destroyed_uniformly_at_the_requested_fraction() {
        let problem = two_pairs();
        let freq = frequencies(&problem, &[0, 1, 0, 1], 0.3, 20_000);
        assert!(freq.iter().all(|f| (f - 0.3).abs() < 0.02), "{freq:?}");
    }
}
