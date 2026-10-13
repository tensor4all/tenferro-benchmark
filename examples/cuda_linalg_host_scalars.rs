//! Public-API reproducer: CUDA-resident input, direct inverse / pseudoinverse.
use tenferro_gpu::cuda::{upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{BackendSessionHost, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0))?;
    let runtime = backend.runtime().clone();
    let host = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])?;
    let input = upload_tensor(&runtime, &host)?;
    let mut failures = 0;
    for operation in ["inv", "pinv_with_rtol"] {
        let result = backend.with_backend_session(|session| {
            if operation == "inv" {
                input.inv(session)
            } else {
                input.pinv_with_rtol(1e-5, session)
            }
        })?;
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
