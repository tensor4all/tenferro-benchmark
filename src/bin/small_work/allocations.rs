//! Separate-build diagnostic: Rust allocator calls on the invoking thread.
//! Foreign BLAS allocations and worker-thread allocations are not counted.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}
struct Counter;
fn record(bytes: usize) {
    ENABLED.with(|enabled| {
        if enabled.get() {
            COUNTS.with(|counts| {
                let (n, b) = counts.get();
                counts.set((n + 1, b + bytes));
            });
        }
    });
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            record(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            record(layout.size());
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = System.realloc(ptr, layout, size);
        if !out.is_null() {
            record(size);
        }
        out
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
pub fn capture<O>(operation: impl FnOnce() -> O) -> (O, (usize, usize)) {
    COUNTS.with(|v| v.set((0, 0)));
    ENABLED.with(|v| v.set(true));
    let output = operation();
    ENABLED.with(|v| v.set(false));
    (output, COUNTS.with(Cell::get))
}
