// Android JavaScript interface types
declare global {
  interface Window {
    ExternalUrlInterface?: {
      openExternalUrl(url: string): void
    }
    SafFileInterface?: {
      startFileSelection(): void
    }
  }
}

export {}
