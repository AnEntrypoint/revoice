use std::sync::Mutex;
use std::time::Instant;

use candle_core::Device;

struct State {
    last: Option<Instant>,
    start: Instant,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn level() -> u8 {
    match std::env::var("RESOUND_PROFILE").as_deref() {
        Ok("3") => 3,
        Ok("2") => 2,
        _ => 0,
    }
}

pub fn tick_at(min_level: u8, device: &Device, label: &str) {
    if level() < min_level {
        return;
    }
    let _ = device.synchronize();
    let now = Instant::now();
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| State {
        last: None,
        start: now,
    });
    let since = match state.last.replace(now) {
        Some(prev) => now.saturating_duration_since(prev),
        None => now.saturating_duration_since(state.start),
    };
    println!(
        "[prof] {label}: {:.1} ms",
        since.as_secs_f64() * 1000.0
    );
}

pub fn tick(device: &Device, label: &str) {
    tick_at(2, device, label)
}
