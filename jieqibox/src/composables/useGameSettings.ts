import { ref, watch } from 'vue'
import { useConfigManager } from './useConfigManager'

// Configuration manager
const configManager = useConfigManager()

/**
 * Get initial settings from the config manager
 * @returns {object} - Object containing initial values for game settings
 */
const getInitialSettings = () => {
  // Only access in client environment
  if (typeof window === 'undefined') {
    return {
      flipMode: 'random' as 'random' | 'free',
      enablePonder: false,
    }
  }

  try {
    const settings = configManager.getGameSettings()
    return {
      flipMode: settings.flipMode || 'random',
      enablePonder: !!settings.enablePonder, // Default to false
    }
  } catch (e) {
    console.error('Failed to get game settings:', e)
    // Return default values on error
    return {
      flipMode: 'random' as 'random' | 'free',
      enablePonder: false,
    }
  }
}

// Create reactive references shared across the application
const { flipMode: initialFlipMode, enablePonder: initialEnablePonder } =
  getInitialSettings()

const flipMode = ref<'random' | 'free'>(initialFlipMode)
const enablePonder = ref<boolean>(initialEnablePonder)

// Flag to track if config is loaded
const isConfigLoaded = ref(false)

// Watch for changes and persist to config file
watch([flipMode, enablePonder], async ([newFlipMode, newEnablePonder]) => {
  // Only save if config is already loaded to avoid overwriting during initialization
  if (!isConfigLoaded.value) return

  const settings = {
    flipMode: newFlipMode,
    enablePonder: newEnablePonder,
  }

  try {
    await configManager.updateGameSettings(settings)
  } catch (error) {
    console.error('Failed to save game settings:', error)
  }
})

// Game settings composable
export function useGameSettings() {
  // Load configuration and update reactive refs
  const loadSettings = async () => {
    try {
      await configManager.loadConfig()
      const settings = configManager.getGameSettings()

      // Update reactive refs
      flipMode.value = settings.flipMode || 'random'
      enablePonder.value = !!settings.enablePonder

      isConfigLoaded.value = true
    } catch (error) {
      console.error('Failed to load game settings:', error)
      isConfigLoaded.value = true // Still mark as loaded to enable saving
    }
  }

  // Initialize settings on first import
  if (typeof window !== 'undefined') {
    loadSettings()
  }

  return {
    flipMode,
    enablePonder,
    loadSettings,
  }
}
