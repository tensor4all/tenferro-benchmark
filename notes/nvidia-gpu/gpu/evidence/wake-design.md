# Wake-on-ready-work repair

The no-sleep counterfactual establishes a useful large-BMM effect but fails nonregression and CPU-efficiency checks. Do not ship it.

Use std::thread::park/unpark in the existing custom channel, retaining its allocation-free task slots and batching. The existing normal_channel alternative boxes every task and changes batching, so switching the whole channel is not a minimal policy repair.

- State owns a OnceLock<Thread>, populated by the server before its first queue check. Full-buffer publication from enqueue and partial-buffer flush both notify when their atomic increment makes enqueued_count reach CHANNEL_MAX_TASK. No notification on incomplete buffers.
- Keep spin/yield budgets; replace only the final timed sleep with park. An unpark token persists if the producer races between the empty queue check and park. Spurious wakeups are harmless because the loop rechecks the queue. No new atomics ordering or task ownership changes.
- Startup notifications may precede Thread registration: the server registers before checking the published queue, so it sees the full batch without needing a token. Existing task execution, buffer swap ordering, scoped completion, panic handling and async retention remain unchanged.
- Test full and partial batches after idle, publication before server startup, and the check-to-park race with notifications arriving before parking. Run the existing channel concurrency/lifetime tests.
- This affects all users of the custom channel, not only CUDA. Numerical/queue tests plus the complete paired performance gates are required; other hardware/provider timing remains unverified.

## Predeclared second experiment

Before wake-candidate execution: use the exact same baseline executable, cases, 1T settings, 3-pair order, 5/30 samples, validity thresholds and acceptance ratios as protocol.md. Candidate now changes only wakeup notification/parking rather than removing server sleep. Results go to wake-results/, retaining the rejected first candidate. Record whole-process CPU usage; it should remain <=1.10 times baseline CPU-seconds in every pair as an additional gate. Collect representative CUDA+OSRT traces separately. No selective retries or threshold relaxation.
