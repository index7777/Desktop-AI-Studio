export type HardwareStatus={gpuName:string;vramTotalMb:number|null;vramFreeMb:number|null;nvidiaAvailable:boolean;engineState:string}; export type GenerateRequest={prompt:string;width:number;height:number;steps:number;seed:number;trueCfgScale:number;negativePrompt?:string|null;inputDataUrl?:string|null}; export type GenerateResult={imageDataUrl:string;path:string;seed:number;elapsedSeconds:number;width:number;height:number};
export async function getHardwareStatus():Promise<HardwareStatus>{if(!("__TAURI_INTERNALS__" in window))return{gpuName:"Browser Preview",vramTotalMb:null,vramFreeMb:null,nvidiaAvailable:false,engineState:"ui-preview"};const{invoke}=await import("@tauri-apps/api/core");return invoke("hardware_status")}
export async function generateImage(request:GenerateRequest):Promise<GenerateResult>{if(!("__TAURI_INTERNALS__" in window))throw new Error("請使用桌面模式執行本機模型。");const{invoke}=await import("@tauri-apps/api/core");return invoke("generate_image",{request})}
export type ModelStatus={id:string;name:string;modelId:string;path:string;installed:boolean;sizeBytes:number};
async function desktopInvoke<T>(command:string):Promise<T>{if(!("__TAURI_INTERNALS__" in window))throw new Error("模型管理需要桌面模式。");const{invoke}=await import("@tauri-apps/api/core");return invoke<T>(command)}
export const getModelStatus=()=>desktopInvoke<ModelStatus>("model_status");
export const installModel=()=>desktopInvoke<ModelStatus>("install_model");
export const removeModel=()=>desktopInvoke<ModelStatus>("remove_model");

export type ModelDownloadProgress={active:boolean;phase:string;downloadedBytes:number;totalBytes:number;percent:number|null;message?:string|null};
export async function onModelDownloadProgress(handler:(progress:ModelDownloadProgress)=>void){if(!("__TAURI_INTERNALS__" in window))return()=>{};const{listen}=await import("@tauri-apps/api/event");return listen<ModelDownloadProgress>("model-download-progress",e=>handler(e.payload))}

export const getModelDownloadStatus=()=>desktopInvoke<ModelDownloadProgress>("model_download_status");

export type EngineProgress={requestId:string;state:string};
export async function onEngineProgress(handler:(progress:EngineProgress)=>void){if(!("__TAURI_INTERNALS__" in window))return()=>{};const{listen}=await import("@tauri-apps/api/event");return listen<EngineProgress>("engine-progress",e=>handler(e.payload))}
