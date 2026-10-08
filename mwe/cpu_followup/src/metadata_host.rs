//! Borrowed Host view arms for #2040. Owners and both view vectors outlive timing.
use super::{black_box, data, json, Instant, Result, Value};
use tenferro_tensor::{
    DynRank, Host, Rank, StridedSliceSpec, TensorRank, TypedTensor, TypedTensorView,
};

fn measure<R: TensorRank, O: TensorRank>(
    op_name: &str,
    runs: usize,
    target: u128,
    make: impl Fn() -> Result<TypedTensor<f64, R, Host>>,
    op: impl for<'a> Fn(&TypedTensorView<'a, f64, R, Host>) -> Result<TypedTensorView<'a, f64, O, Host>>,
) -> Result<Value> {
    // Explicit initialization and validation precede even zero-sample execution.
    let owner = make()?;
    let input = owner.as_view();
    let first = op(&input)?;
    let signature = json!({"kind":"metadata","shape":first.shape(),"strides":first.strides(),"offset":first.offset(),"metadata_only":true});
    for i in 0..first.shape().iter().product::<usize>() {
        let (index, physical) = match op_name {
            "reshape_view" => (vec![i % 32, i / 32], i),
            "slice_view" => (vec![i], 128 + 2 * i),
            "transpose_view" => (vec![i % 2, i / 2], i / 2 + 2 * (i % 2)),
            _ => unreachable!(),
        };
        let actual = first.get(&index).ok_or("missing view element")?;
        let expected = &owner.as_slice()[physical];
        if actual != expected || !std::ptr::eq(actual, expected) {
            return Err("view payload or owner alias differs".into());
        }
    }
    let bytes = owner.as_slice().len() * 8
        + std::mem::size_of::<TypedTensor<f64, R, Host>>()
        + std::mem::size_of::<TypedTensorView<'_, f64, R, Host>>()
        + std::mem::size_of::<TypedTensorView<'_, f64, O, Host>>()
        + 256;
    drop(first);
    drop(input);
    drop(owner);
    let cap = (512 * 1024 * 1024 / bytes.max(1)).clamp(1, 2_000_000);
    let batch = |count: usize| -> Result<u128> {
        let owners = (0..count).map(|_| make()).collect::<Result<Vec<_>>>()?;
        let inputs = owners
            .iter()
            .map(|owner| owner.as_view())
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(count);
        let start = Instant::now();
        for input in &inputs {
            outputs.push(op(input)?);
        }
        let elapsed = start.elapsed().as_nanos();
        black_box(&outputs);
        black_box(&owners);
        drop(outputs);
        drop(inputs);
        drop(owners);
        Ok(elapsed)
    };
    let mut samples = Vec::new();
    let mut count = 0;
    let mut calibrated = 0;
    if runs > 0 {
        for _ in 0..3 {
            batch(1)?;
        }
        count = 1;
        loop {
            calibrated = batch(count)?;
            if calibrated >= target || count == cap {
                break;
            }
            count = (count * 2).min(cap);
        }
        for i in 0..runs {
            samples.push(json!({"sample_index":i,"iterations":count,"elapsed_ns":batch(count)?}));
        }
    }
    Ok(json!({"samples":samples,"outputs":[signature],
        "calibration":{"iterations":count,"elapsed_ns":calibrated,"target_ns":target,"memory_cap_bytes":512*1024*1024},
        "lifetime_contract":"borrowed input and output views; separate owner per operation retained until after timing and output destruction",
        "scope_note":"pure metadata, no session; Host owners and as_view inputs constructed before every timer; payload and pointer alias validated untimed",
        "input_rank":std::any::type_name::<R>(),"output_rank":std::any::type_name::<O>(),"representation":"TypedTensorView<f64,_,Host>"}))
}

pub(super) fn run(case: &Value, path: &str, runs: usize, target: u128) -> Result<Value> {
    let name = case["op"].as_str().unwrap();
    let slice = [StridedSliceSpec::new(128, Some(3968), 2)];
    macro_rules! run_arm {
        ($rank:ty, $shape:expr, $out:ty, $call:expr) => {
            measure::<$rank, $out>(
                name,
                runs,
                target,
                || {
                    Ok(TypedTensor::<f64, $rank, Host>::from_host_vec_col_major(
                        $shape,
                        data(
                            if name == "transpose_view" {
                                4
                            } else if name == "reshape_view" {
                                1024
                            } else {
                                4096
                            },
                            1,
                        ),
                    )?)
                },
                $call,
            )
        };
    }
    match (path, name) {
        ("metadata-host-dyn", "reshape_view") => run_arm!(DynRank, vec![1024], DynRank, |v| Ok(
            v.reshape_view(&[32, 32])?
        )),
        ("metadata-host-dyn", "transpose_view") => run_arm!(DynRank, vec![2, 2], DynRank, |v| Ok(
            v.transpose_view([1, 0])?
        )),
        ("metadata-host-dyn", "slice_view") => {
            run_arm!(DynRank, vec![4096], DynRank, |v| Ok(v.slice_view(&slice)?))
        }
        ("metadata-host-static", "reshape_view") => {
            run_arm!(Rank<1>, [1024], DynRank, |v| Ok(v.reshape_view(&[32, 32])?))
        }
        ("metadata-host-static", "transpose_view") => {
            run_arm!(Rank<2>, [2, 2], Rank<2>, |v| Ok(v.transpose_view([1, 0])?))
        }
        ("metadata-host-static", "slice_view") => {
            run_arm!(Rank<1>, [4096], Rank<1>, |v| Ok(v.slice_view(&slice)?))
        }
        _ => Err("unknown typed metadata arm".into()),
    }
}
