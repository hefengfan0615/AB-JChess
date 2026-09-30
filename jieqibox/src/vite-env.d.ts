/// <reference types="vite/client" />

// Tauri API type definitions for Android UCI engine support
declare module '@tauri-apps/api/core' {
  export function invoke<T = any>(
    cmd: string,
    args?: Record<string, any>
  ): Promise<T>
}

// Android-specific types
interface AndroidEngineInfo {
  defaultPath: string
  isExecutable: boolean
  error?: string
}

// Platform detection
declare global {
  interface Window {
    __TAURI__?: {
      platform: string
    }
  }
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}
