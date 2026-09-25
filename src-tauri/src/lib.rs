use serde::Serialize;
use std::process::Command;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HardwareStatus {
    gpu_name: String,
    vram_total_mb: Option<u64>,
    vram_free_mb: Option<u64>,
    nvidia_available: bool,
    engine_state: String,
}

#[tauri::command]
fn hardware_status() -> HardwareStatus {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = text.lines().next() {
                let parts: Vec<_> = line.split(',').map(str::trim).collect();
                return HardwareStatus {
                    gpu_name: parts.first().unwrap_or(&"NVIDIA GPU").to_string(),
                    vram_total_mb: parts.get(1).and_then(|v| v.parse().ok()),
                    vram_free_mb: parts.get(2).and_then(|v| v.parse().ok()),
                    nvidia_available: true,
                    engine_state: "runtime-ready".into(),
                };
            }
        }
    }

    HardwareStatus {
        gpu_name: "未偵測到 NVIDIA GPU".into(),
        vram_total_mb: None,
        vram_free_mb: None,
        nvidia_available: false,
        engine_state: "hardware-unavailable".into(),
    }
}

#[tauri::command]
fn app_version() -> &'static str { env!("CARGO_PKG_VERSION") }

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![hardware_status, app_version])
        .run(tauri::generate_context!())
        .expect("error while running AI Studio");
}
