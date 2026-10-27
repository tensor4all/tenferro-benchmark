// Temporary diagnostic: semantic graph inventory and empty executor handoff.
use std::{hint::black_box, time::Instant};
use tenferro_ad::AdContext;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, TracedTensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
        .build()?;
    let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.2, 0.2, 4.0])?;
    let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
    let (q, r) = a.qr()?;
    let loss = (&q.reduce_sum(Some(&[0, 1]))? + &r.reduce_sum(Some(&[0, 1]))?)?;
    let qr_grad = ad.grad(&loss, &a)?;
    let x = a.solve(&b)?;
    let solve_loss = x.reduce_sum(Some(&[0, 1]))?;
    let ga = ad.grad(&solve_loss, &a)?;
    let gb = ad.grad(&solve_loss, &b)?;
    for (label, outs) in [
        ("qr_vjp", vec![&loss, &qr_grad]),
        ("solve", vec![&x]),
        ("solve_backward", vec![&ga, &gb, &solve_loss]),
    ] {
        let p = GraphCompiler::new().compile_many(&outs)?;
        println!("GRAPH {label} {} ops", p.program().operations().len());
        for o in p.program().operations() {
            println!("{:?}", o.op());
        }
    }
    for threads in [1, 4] {
        // tenferro-rs #2206 removed the shared CPU context; the equivalent
        // empty placement probe uses the public external-provider entry.
        let ctx = tenferro_cpu::Pool::builder().threads(threads).build()?;
        let mut exec = tenferro_cpu::threadteam::Exec::new(&ctx, threads)?;
        for _ in 0..100 {
            black_box(exec.reborrow().external(|_| black_box(1)));
        }
        let mut times = Vec::new();
        for _ in 0..1001 {
            let start = Instant::now();
            let out = exec.reborrow().external(|_| black_box(1));
            times.push(start.elapsed().as_nanos());
            black_box(out);
        }
        times.sort();
        println!("EMPTY_EXTERNAL threads={threads} median_ns={}", times[500]);
    }
    Ok(())
}
