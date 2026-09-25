export type HardwareStatus = {
  gpuName: string;
  vramTotalMb: number | null;
  vramFreeMb: number | null;
  nvidiaAvailable: boolean;
  engineState: string;
};

const browserFallback: HardwareStatus = {
  gpuName: "Browser Preview",
  vramTotalMb: null,
  vramFreeMb: null,
  nvidiaAvailable: false,
  engineState: "ui-preview",
};

export async function getHardwareStatus(): Promise<HardwareStatus> {
  if (!("__TAURI_INTERNALS__" in window)) return browserFallback;
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<HardwareStatus>("hardware_status");
}
