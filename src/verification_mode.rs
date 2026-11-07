//! How a suite problem's numbers are checked.
//!
//! A problem's `verify.mode` names the check that backs its row:
//!
//! * `cpu_reference` (default, and the only mode that existed before) re-runs
//!   the same compiled graph on a CPU engine and compares every output element.
//! * `analytic` compares every output element against a closed-form reference
//!   (for example `d sum(svd(A)) / dA = U V^T`), so the cost is the reference
//!   itself rather than a second AD pass.
//! * `finite_difference` compares the directional derivative the row reports
//!   against a central difference of the primal loss.
//! * `measure_only` runs no check at all.
//!
//! `measure_only` is the only mode that removes evidence, so it is refused
//! unless the same `verify` block states a non-empty `reason`. An unexplained
//! measurement-only row reads as verified to every consumer downstream, which
//! is the one failure this contract exists to make impossible.

use serde_yaml::Value;

/// The mode a `verify` block declares when it declares none.
pub const DEFAULT_MODE: &str = "cpu_reference";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationMode {
    CpuReference,
    Analytic,
    FiniteDifference,
    MeasureOnly,
}

impl VerificationMode {
    /// The word the manifest and the result records use for this mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuReference => "cpu_reference",
            Self::Analytic => "analytic",
            Self::FiniteDifference => "finite_difference",
            Self::MeasureOnly => "measure_only",
        }
    }

    /// Parse a manifest word, or `None` when it names no mode.
    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "cpu_reference" => Some(Self::CpuReference),
            "analytic" => Some(Self::Analytic),
            "finite_difference" => Some(Self::FiniteDifference),
            "measure_only" => Some(Self::MeasureOnly),
            _ => None,
        }
    }

    pub fn is_measure_only(self) -> bool {
        self == Self::MeasureOnly
    }
}

fn problem_id(problem: &Value) -> &str {
    problem["id"].as_str().unwrap_or("<problem>")
}

fn mode_word<'a>(problem: &'a Value, defaults: &'a Value) -> &'a str {
    problem["verify"]["mode"]
        .as_str()
        .or_else(|| defaults["verify"]["mode"].as_str())
        .unwrap_or(DEFAULT_MODE)
}

/// The reason a `measure_only` problem gives for carrying no check.
///
/// The problem's own `reason` wins over the suite default so a case that
/// overrides the mode can also override the explanation.
pub fn measure_only_reason(problem: &Value, defaults: &Value) -> Option<String> {
    problem["verify"]["reason"]
        .as_str()
        .or_else(|| defaults["verify"]["reason"].as_str())
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
        .map(str::to_string)
}

/// The verification mode of one problem, with the suite defaults filled in.
///
/// Fails on an unknown mode and on `measure_only` without a non-empty reason,
/// so a manifest cannot buy a cheaper run by dropping its evidence.
pub fn mode_for(problem: &Value, defaults: &Value) -> Result<VerificationMode, String> {
    let word = mode_word(problem, defaults);
    let mode = VerificationMode::parse(word).ok_or_else(|| {
        format!(
            "{}: unknown verify.mode={word:?}; expected one of cpu_reference, analytic, \
             finite_difference, measure_only",
            problem_id(problem)
        )
    })?;
    if mode.is_measure_only() && measure_only_reason(problem, defaults).is_none() {
        return Err(format!(
            "{}: verify.mode=measure_only requires a non-empty verify.reason; \
             a measurement with no check must say why it has none",
            problem_id(problem)
        ));
    }
    Ok(mode)
}

/// A copy of `problem` whose `verify` block carries the validated mode and the
/// `measure_only` reason, so every runner reads one resolved contract.
///
/// Only `mode` and `reason` are written: `reference`, `rtol` and `atol` keep
/// whatever the problem declared, and a problem that declared no `verify`
/// block still resolves to `cpu_reference` with its previous tolerances.
pub fn resolve(problem: &Value, defaults: &Value) -> Result<Value, String> {
    let mode = mode_for(problem, defaults)?;
    let mut resolved = problem.clone();
    let mapping = resolved
        .as_mapping_mut()
        .ok_or_else(|| format!("{}: problem is not a mapping", problem_id(problem)))?;
    let verify = mapping
        .entry(Value::from("verify"))
        .or_insert_with(|| Value::Mapping(Default::default()));
    if !verify.is_mapping() {
        *verify = Value::Mapping(Default::default());
    }
    let verify = verify.as_mapping_mut().expect("verify block is a mapping");
    verify.insert(Value::from("mode"), Value::from(mode.as_str()));
    if let Some(reason) = measure_only_reason(problem, defaults) {
        verify.insert(Value::from("reason"), Value::from(reason));
    }
    Ok(resolved)
}

/// The mode of a problem that already went through [`resolve`].
pub fn resolved_mode(problem: &Value) -> Option<VerificationMode> {
    problem["verify"]["mode"].as_str().and_then(VerificationMode::parse)
}

/// The recorded `measure_only` reason of a resolved problem.
pub fn resolved_measure_only_reason(problem: &Value) -> Option<&str> {
    problem["verify"]["reason"]
        .as_str()
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem(yaml: &str) -> Value {
        serde_yaml::from_str(yaml).expect("test problem")
    }

    #[test]
    fn absent_verify_block_is_cpu_reference() {
        let defaults = problem("{}");
        let case = problem("{id: case_a, op: matmul}");
        assert_eq!(mode_for(&case, &defaults).unwrap(), VerificationMode::CpuReference);
        let resolved = resolve(&case, &defaults).unwrap();
        assert_eq!(resolved_mode(&resolved), Some(VerificationMode::CpuReference));
        // Resolving a problem with no verify block must not invent tolerances.
        assert!(resolved["verify"]["rtol"].is_null());
    }

    #[test]
    fn suite_default_mode_applies_when_the_problem_declares_none() {
        let defaults = problem("{verify: {mode: analytic, rtol: 1.0e-5, atol: 1.0e-8}}");
        let case = problem("{id: case_a, verify: {rtol: 1.0e-6}}");
        let resolved = resolve(&case, &defaults).unwrap();
        assert_eq!(resolved_mode(&resolved), Some(VerificationMode::Analytic));
        assert_eq!(resolved["verify"]["rtol"].as_f64(), Some(1.0e-6));
    }

    #[test]
    fn a_problem_overrides_the_suite_default_mode() {
        let defaults = problem("{verify: {mode: analytic, rtol: 1.0e-5, atol: 1.0e-8}}");
        let case = problem("{id: case_a, verify: {mode: finite_difference, reason: 'suite default is analytic'}}");
        assert_eq!(mode_for(&case, &defaults).unwrap(), VerificationMode::FiniteDifference);
    }

    #[test]
    fn measure_only_without_a_reason_is_refused() {
        let defaults = problem("{}");
        for block in ["{id: p, verify: {mode: measure_only}}",
                      "{id: p, verify: {mode: measure_only, reason: ''}}",
                      "{id: p, verify: {mode: measure_only, reason: '   '}}"] {
            let case = problem(block);
            let error = mode_for(&case, &defaults).expect_err("must be refused");
            assert!(error.contains("measure_only requires a non-empty verify.reason"), "{error}");
            assert!(resolve(&case, &defaults).is_err());
        }
    }

    #[test]
    fn measure_only_with_a_reason_resolves_and_keeps_it() {
        let defaults = problem("{}");
        let case = problem("{id: p, verify: {mode: measure_only, reason: 'n=4096 reference is unaffordable'}}");
        let resolved = resolve(&case, &defaults).unwrap();
        assert_eq!(resolved_mode(&resolved), Some(VerificationMode::MeasureOnly));
        assert_eq!(
            resolved_measure_only_reason(&resolved),
            Some("n=4096 reference is unaffordable")
        );
    }

    #[test]
    fn unknown_mode_is_refused() {
        let defaults = problem("{}");
        let case = problem("{id: p, verify: {mode: cpu_fp64, rtol: 1.0e-5, atol: 1.0e-8}}");
        let error = mode_for(&case, &defaults).expect_err("must be refused");
        assert!(error.contains("unknown verify.mode"), "{error}");
    }

    #[test]
    fn every_mode_word_round_trips() {
        for mode in [
            VerificationMode::CpuReference,
            VerificationMode::Analytic,
            VerificationMode::FiniteDifference,
            VerificationMode::MeasureOnly,
        ] {
            assert_eq!(VerificationMode::parse(mode.as_str()), Some(mode));
        }
        assert_eq!(VerificationMode::parse(DEFAULT_MODE), Some(VerificationMode::CpuReference));
    }
}
