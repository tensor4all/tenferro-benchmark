//! Public-API reproducer: CUDA-resident input, direct inverse / pseudoinverse.
use tenferro_gpu::cuda::{upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // The caller holds the CUDA engine (tenferro-rs #2206) and passes it to
    // every operation; transfer entry points use the engine's runtime.
    let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0))?;
    let runtime = engine.runtime().clone();
    let host = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])?;
    let input = upload_tensor(&runtime, &host)?;
    let mut failures = 0;
    for operation in ["inv", "pinv_with_rtol"] {
        let result = if operation == "inv" {
            input.inv(&mut engine)
        } else {
            input.pinv_with_rtol(1e-5, &mut engine)
        };
        match result {
            Ok(output) => {
                runtime.synchronize()?;
                println!("{operation}: OK, shape={:?}", output.shape());
            }
            Err(error) => {
                failures += 1;
                eprintln!("{operation}: {error}");
            }
        }
    }
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
