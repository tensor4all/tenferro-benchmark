//! Cheap invariants of an expected output, computed once when fixtures are prepared.
//!
//! The GPU public API harness stores these next to every expected output, and each arm
//! records whether the output it produced still has the same invariants. They are
//! *diagnostics*: the elementwise comparison against the reference bytes decides whether
//! an output is correct, and always runs. No aggregate summary of this shape can stand in
//! for that comparison, because every invariant here is a sum, and one large element can
//! pay the whole allowance a sum is compared with.
//!
//! The invariants are:
//!
//! * `sum` — one signed sum, per component for complex values;
//! * `sum_abs` — the L1 mass, which grows with every element and cannot cancel;
//! * `sum_squares` — the L2 mass;
//! * `max_abs` — the peak magnitude;
//! * `dot_seeded` — a dot product against a fixed, seeded weight vector.
//!
//! `sum`, `sum_abs`, `sum_squares` and `max_abs` are invariant under any
//! permutation, so the weighted dot product is the invariant that separates a
//! transposed or permuted output from a correct one.

/// Version of the stored fingerprint document.
pub const FINGERPRINT_SCHEMA_VERSION: u32 = 1;

/// Five invariants of one expected output.
///
/// Complex values keep the real and imaginary parts of the signed sums separate;
/// the magnitude-based invariants are real by construction.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Fingerprint {
    /// Signed sum of the real parts.
    pub sum_re: f64,
    /// Signed sum of the imaginary parts (always 0 for a real dtype).
    pub sum_im: f64,
    /// Sum of `|z|` over every element.
    pub sum_abs: f64,
    /// Sum of `|z|²` over every element.
    pub sum_squares: f64,
    /// Largest `|z|`.
    pub max_abs: f64,
    /// Dot product against the seeded weight vector, real part.
    pub dot_seeded_re: f64,
    /// Dot product against the seeded weight vector, imaginary part.
    pub dot_seeded_im: f64,
    /// Number of elements the invariants cover.
    pub elements: u64,
}

impl Fingerprint {
    /// Compute the invariants of one column-major output.
    ///
    /// `imaginary` is empty for a real dtype and one value per element for a complex
    /// dtype.
    pub fn of(real: &[f64], imaginary: &[f64]) -> Self {
        assert!(
            imaginary.is_empty() || imaginary.len() == real.len(),
            "imaginary part must be empty or as long as the real part"
        );
        let complex = !imaginary.is_empty();
        let mut seeded = SeededWeights::default();
        let mut fingerprint = Fingerprint {
            sum_re: 0.0,
            sum_im: 0.0,
            sum_abs: 0.0,
            sum_squares: 0.0,
            max_abs: 0.0,
            dot_seeded_re: 0.0,
            dot_seeded_im: 0.0,
            elements: real.len() as u64,
        };
        for (index, &re) in real.iter().enumerate() {
            let im = if complex { imaginary[index] } else { 0.0 };
            let magnitude = re.hypot(im);
            let weight = seeded.next_weight();
            fingerprint.sum_re += re;
            fingerprint.sum_im += im;
            fingerprint.sum_abs += magnitude;
            fingerprint.sum_squares += re * re + im * im;
            fingerprint.max_abs = fingerprint.max_abs.max(magnitude);
            fingerprint.dot_seeded_re += weight * re;
            fingerprint.dot_seeded_im += weight * im;
        }
        fingerprint
    }

    /// Whether `actual` agrees with this fingerprint within the case's tolerance.
    ///
    /// A diagnostic agreement test, never an acceptance test. `atol` and `rtol` are the
    /// same elementwise tolerances the reference comparison uses, applied to a sum instead
    /// of an element: an absolute per-element allowance scaled by the element count, plus
    /// a relative allowance scaled by the mass the invariant measures. An output that is
    /// closed in this sense can still be elementwise wrong — one large element pays the
    /// whole allowance — so the caller compares every element whatever this returns.
    ///
    /// `self` is the stored (expected) fingerprint: every allowance scales from it.
    pub fn matches(&self, actual: &Fingerprint, atol: f64, rtol: f64) -> bool {
        if actual.elements != self.elements {
            return false;
        }
        let count = self.elements as f64;
        let absolute = atol * count;
        let mass = absolute + rtol * self.sum_abs;
        let peaks = 2.0 * self.max_abs * mass;
        let close = |actual: f64, expected: f64, scale: f64| {
            (actual - expected).abs() <= scale && (actual - expected).abs().is_finite()
        };
        close(actual.sum_re, self.sum_re, mass)
            && close(actual.sum_im, self.sum_im, mass)
            && close(actual.sum_abs, self.sum_abs, mass)
            && close(
                actual.sum_squares,
                self.sum_squares,
                absolute + rtol * self.sum_squares + peaks,
            )
            && close(actual.max_abs, self.max_abs, atol + rtol * self.max_abs)
            && close(actual.dot_seeded_re, self.dot_seeded_re, mass)
            && close(actual.dot_seeded_im, self.dot_seeded_im, mass)
    }
}

/// The fixed, seeded weight vector, one weight per element in file order.
///
/// A splitmix64 sequence seeded with a constant: identical in the Rust harness and
/// in the fixture writer, so a stored fingerprint and a live one use the same vector.
/// `scripts/benchmark_gpu_public_api.py` reimplements this step-for-step; the two are
/// pinned together by a shared constant in the tests of both.
///
/// Each weight is a real value in `[-1, 1)` taken from 53 bits of the stream, not a `±1`
/// sign. A sign vector gives half of all element pairs the same weight, and the weighted
/// dot product cannot see a swap of two elements that share a weight: with `±1` weights
/// `[1, 2, 3, 4]` and `[1, 3, 2, 4]` had the same dot product, so the one invariant that
/// is supposed to separate orderings separated nothing.
#[derive(Default)]
pub struct SeededWeights {
    state: u64,
}

impl SeededWeights {
    /// The sequence seed, shared with the fixture writer.
    pub const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Return the next weight in `[-1, 1)`.
    pub fn next_weight(&mut self) -> f64 {
        self.state = self.state.wrapping_add(Self::SEED);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53 bits scaled into [-1, 1): exact in f64, and |weight| <= 1 keeps the
        // deviation bound in `matches` a valid one for a weighted sum.
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// One stored fingerprint document, written next to its fixture by the preparation
/// step and read by every arm.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StoredFingerprint {
    /// Document version; a reader refuses a version it does not know.
    pub schema_version: u32,
    /// Case the output belongs to.
    pub case_id: String,
    /// Index of the output within the case.
    pub output: usize,
    /// Dtype name the harness uses (`f64`, `c64`, ...).
    pub dtype: String,
    /// Column-major shape.
    pub shape: Vec<usize>,
    /// The invariants, or `None` when the expected values are not all finite.
    ///
    /// `-inf` is a legitimate expected value (`masked_log_softmax` produces it for a
    /// masked element), and its invariants are not representable in JSON. The document
    /// then says why instead of carrying an unparseable number, and the arm records no
    /// fingerprint diagnostic for that output.
    #[serde(default)]
    pub fingerprint: Option<Fingerprint>,
    /// Why the invariants are absent, when they are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
}

impl StoredFingerprint {
    /// File name a fixture's fingerprint is stored under, next to the fixture.
    pub fn file_name(fixture_file: &str) -> String {
        format!("{fixture_file}.fingerprint.json")
    }

    /// Parse a stored document, refusing an unknown version.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let stored: Self = serde_json::from_slice(bytes)
            .map_err(|source| format!("fingerprint json: {source}"))?;
        if stored.schema_version != FINGERPRINT_SCHEMA_VERSION {
            return Err(format!(
                "fingerprint {} has schema version {}, expected {}",
                stored.case_id, stored.schema_version, FINGERPRINT_SCHEMA_VERSION
            ));
        }
        Ok(stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-language reference: the same array and the same constants are asserted by
    /// `scripts/ci/tests/test_fingerprint_parity.py`, so the Rust and Python
    /// implementations cannot drift apart silently.
    const REFERENCE_REAL: [f64; 6] = [1.0, -2.5, 0.0, 4.0, -0.5, 3.0];
    const REFERENCE_IMAGINARY: [f64; 6] = [0.5, 0.0, -1.0, 2.0, 0.0, -0.25];
    const REFERENCE: Fingerprint = Fingerprint {
        sum_re: 5.0,
        sum_im: 1.25,
        sum_abs: 12.600_568_588_447_548,
        sum_squares: 37.812_5,
        max_abs: 4.472_135_954_999_58,
        dot_seeded_re: 4.233_645_350_156_905,
        dot_seeded_im: 3.300_308_295_534_698_3,
        elements: 6,
    };

    #[test]
    fn invariants_of_the_cross_language_reference_match() {
        let fingerprint = Fingerprint::of(&REFERENCE_REAL, &REFERENCE_IMAGINARY);
        assert_eq!(fingerprint.elements, 6);
        assert!((fingerprint.sum_re - REFERENCE.sum_re).abs() < 1e-12);
        assert!((fingerprint.sum_im - REFERENCE.sum_im).abs() < 1e-12);
        assert!((fingerprint.sum_abs - REFERENCE.sum_abs).abs() < 1e-12);
        assert!((fingerprint.sum_squares - REFERENCE.sum_squares).abs() < 1e-12);
        assert!((fingerprint.max_abs - REFERENCE.max_abs).abs() < 1e-12);
        assert!((fingerprint.dot_seeded_re - REFERENCE.dot_seeded_re).abs() < 1e-12);
        assert!((fingerprint.dot_seeded_im - REFERENCE.dot_seeded_im).abs() < 1e-12);
    }

    #[test]
    fn seeded_weights_are_stable_and_bounded() {
        let mut weights = SeededWeights::default();
        let taken: Vec<f64> = (0..8).map(|_| weights.next_weight()).collect();
        // Pinned so a change to the sequence cannot pass unnoticed on either side.
        assert_eq!(
            taken,
            [
                0.766_621_616_427_285_2,
                -0.136_944_005_902_980_06,
                -0.947_132_456_814_804_5,
                0.941_763_956_307_657,
                -0.787_306_616_865_575_1,
                -0.345_348_471_563_748_5,
                -0.652_264_268_080_634_3,
                0.543_093_112_663_134,
            ],
            "the seeded vector is shared with the fixture writer"
        );
        assert!(taken.iter().all(|w| w.abs() <= 1.0));
        let mut fresh = SeededWeights::default();
        let again: Vec<f64> = (0..8).map(|_| fresh.next_weight()).collect();
        assert_eq!(taken, again);
    }

    #[test]
    fn a_permutation_is_caught_by_the_seeded_dot_product() {
        let expected = Fingerprint::of(&[1.0, 2.0, 3.0, 4.0], &[]);
        let mut permuted = [1.0, 2.0, 3.0, 4.0];
        permuted.swap(1, 3);
        let actual = Fingerprint::of(&permuted, &[]);
        // Every magnitude invariant is blind to the permutation.
        assert!((actual.sum_re - expected.sum_re).abs() < 1e-12);
        assert!((actual.sum_abs - expected.sum_abs).abs() < 1e-12);
        assert!((actual.sum_squares - expected.sum_squares).abs() < 1e-12);
        assert!((actual.max_abs - expected.max_abs).abs() < 1e-12);
        assert!(!expected.matches(&actual, 1e-9, 1e-9));
    }

    #[test]
    fn adjacent_permutations_differ() {
        // The `±1` weights this replaced were [-1, +1, +1, -1] for these four
        // elements, so both arrays had dot product 0 and every other invariant was
        // equal: a swap of positions 1 and 2 was invisible.
        let expected = Fingerprint::of(&[1.0, 2.0, 3.0, 4.0], &[]);
        let actual = Fingerprint::of(&[1.0, 3.0, 2.0, 4.0], &[]);
        assert!((actual.sum_abs - expected.sum_abs).abs() < 1e-12);
        assert!((actual.sum_squares - expected.sum_squares).abs() < 1e-12);
        assert!((actual.max_abs - expected.max_abs).abs() < 1e-12);
        assert!((actual.dot_seeded_re - expected.dot_seeded_re).abs() > 1e-3);
        assert!(!expected.matches(&actual, 1e-9, 1e-9));
    }

    #[test]
    fn small_roundoff_still_matches() {
        let expected = Fingerprint::of(&[1.0, -2.0, 3.5], &[]);
        let jittered = [1.0 + 1e-12, -2.0 - 1e-12, 3.5 + 1e-12];
        assert!(expected.matches(&Fingerprint::of(&jittered, &[]), 1e-9, 1e-9));
    }

    #[test]
    fn a_different_element_count_never_matches() {
        let expected = Fingerprint::of(&[1.0, 2.0], &[]);
        let actual = Fingerprint::of(&[1.0, 2.0, 0.0], &[]);
        assert!(!expected.matches(&actual, 1e-9, 1e-9));
    }

    #[test]
    fn a_stored_document_round_trips_and_refuses_another_version() {
        let stored = StoredFingerprint {
            schema_version: FINGERPRINT_SCHEMA_VERSION,
            case_id: "cholesky_f64_32x32".into(),
            output: 0,
            dtype: "f64".into(),
            shape: vec![32, 32],
            fingerprint: Some(Fingerprint::of(&[1.0, 2.0], &[])),
            unavailable_reason: None,
        };
        let bytes = serde_json::to_vec(&stored).unwrap();
        assert_eq!(
            StoredFingerprint::from_bytes(&bytes).unwrap().case_id,
            stored.case_id
        );

        let mut newer: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        newer["schema_version"] = serde_json::json!(FINGERPRINT_SCHEMA_VERSION + 1);
        let error =
            StoredFingerprint::from_bytes(&serde_json::to_vec(&newer).unwrap()).unwrap_err();
        assert!(error.contains("schema version"), "{error}");
    }

    #[test]
    fn a_real_dtype_treats_missing_imaginary_parts_as_zero() {
        let stored = StoredFingerprint {
            schema_version: FINGERPRINT_SCHEMA_VERSION,
            case_id: "case".into(),
            output: 0,
            dtype: "f64".into(),
            shape: vec![3],
            fingerprint: Some(Fingerprint::of(&[1.0, 2.0, 3.0], &[])),
            unavailable_reason: None,
        };
        let bytes = serde_json::to_vec(&stored).unwrap();
        let read = StoredFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(read.fingerprint.unwrap().sum_im, 0.0);
    }

    #[test]
    fn a_document_without_invariants_parses_and_says_why() {
        // `masked_log_softmax` writes `-inf` for a masked element, so its invariants are
        // not representable in JSON. The document must still parse, or the arm would
        // fail that case before the elementwise comparison ever ran.
        let stored = StoredFingerprint {
            schema_version: FINGERPRINT_SCHEMA_VERSION,
            case_id: "masked_log_softmax_f32_n1024".into(),
            output: 0,
            dtype: "f32".into(),
            shape: vec![32, 32],
            fingerprint: None,
            unavailable_reason: Some("expected values are not all finite".into()),
        };
        let bytes = serde_json::to_vec(&stored).unwrap();
        let read = StoredFingerprint::from_bytes(&bytes).unwrap();
        assert!(read.fingerprint.is_none());
        assert!(read
            .unavailable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("not all finite")));
    }

    #[test]
    fn fingerprint_files_sit_next_to_their_fixture() {
        assert_eq!(
            StoredFingerprint::file_name("cholesky_f64_32x32_out0.bin"),
            "cholesky_f64_32x32_out0.bin.fingerprint.json"
        );
    }
}
