use super::*;
pub fn run(case:&Value,path:&str,threads:usize,runs:usize,target:u128)->Result<Value>{
 let patterns:Value=serde_json::from_str(include_str!("../../../data/instances/permutation_patterns.json"))?;
 let p=patterns["patterns"].as_array().unwrap().iter().find(|p|p["id"]==case["op"]).ok_or("pattern missing")?;
 let shape:Vec<usize>=serde_json::from_value(p["shape"].clone())?;
 let perm:Vec<usize>=serde_json::from_value(p["perm"].clone())?;
 let mut stride=1isize;let default:Vec<_>=shape.iter().map(|&n|{let s=stride;stride*=n as isize;s}).collect();
 let strides:Vec<isize>=if p["src_layout"]["kind"]=="col_major"{default}else{serde_json::from_value(p["src_layout"]["strides"].clone())?};
 let outshape:Vec<_>=perm.iter().map(|&i|shape[i]).collect();let n:usize=shape.iter().product();
 let source:Vec<f64>=(0..n).map(|i|i as f64+1.).collect();
 let view=TypedTensorView::<f64>::from_slice(shape.clone(),strides.clone(),0,&source)?.transpose_view(&perm)?;
 let mut row=if path=="strided-rs"||path.starts_with("strided-") {
   let array=strided_view::StridedArray::from_parts(source.clone(),&shape,&strides,0)?;
   let src=array.view().permute(&perm)?;
   let mut call=||->Result<_>{let mut dst=strided_view::StridedArray::<f64>::col_major(&outshape);
    if threads==1 {strided_perm::copy_into_col_major(&mut dst.view_mut(),&src)?;}else{strided_perm::copy_into_col_major_par(&mut dst.view_mut(),&src)?;}Ok(dst)};
   let first=call()?;let t=Tensor::from_vec_col_major(outshape.clone(),first.data().to_vec())?;let sig=signature(&t)?;drop(t);drop(first);
   let mut r=sample(n*8,runs,target,units,|()|call())?;r["outputs"]=json!([sig]);r
 }else{
   let mut b=CpuBackend::with_threads(threads)?;
   b.with_backend_session(|s|->Result<Value>{tensor_measure(n*8,runs,target,|count|Ok((0..count).map(|_|TensorRead::from_view(TensorView::F64(view.clone()))).collect()),|r|Ok(s.to_contiguous_read(r)?))})??
 };
 row["shape"]=json!(shape);row["permutation"]=json!(perm);row["reference_policy"]=json!("preselected col-major copy; serial at 1T, parallel at 4T; fresh destination allocation");Ok(row)
}
