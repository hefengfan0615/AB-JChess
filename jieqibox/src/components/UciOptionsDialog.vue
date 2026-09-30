<template>
  <v-dialog
    v-model="isVisible"
    :max-width="isMobile ? '95vw' : '800px'"
    persistent
  >
    <v-card class="uci-options-card">
      <v-card-title class="dialog-title">
        <span class="title-text">{{ $t('uciOptions.title') }}</span>
        <v-spacer></v-spacer>
        <v-btn icon @click="closeDialog" class="close-btn">
          <v-icon :color="isDark ? 'white' : 'black'">mdi-close</v-icon>
        </v-btn>
      </v-card-title>

      <v-card-text class="options-container">
        <div v-if="isLoading" class="loading-section">
          <v-progress-circular
            indeterminate
            color="primary"
            size="48"
          ></v-progress-circular>
          <span class="loading-text">{{ $t('uciOptions.loadingText') }}</span>
        </div>

        <div v-else-if="!isEngineLoaded" class="empty-section">
          <v-icon size="64" color="grey">mdi-engine-off</v-icon>
          <p class="empty-text">{{ $t('uciOptions.noEngineLoaded') }}</p>
          <p class="empty-text-secondary">
            {{ $t('uciOptions.pleaseLoadEngineFirst') }}
          </p>
        </div>

        <div v-else-if="uciOptions.length === 0" class="empty-section">
          <v-icon size="64" color="grey">mdi-cog-off</v-icon>
          <p class="empty-text">{{ $t('uciOptions.noOptionsAvailable') }}</p>
          <v-btn
            color="primary"
            @click="refreshOptions"
            size="large"
            class="action-btn"
          >
            {{ $t('uciOptions.refreshOptions') }}
          </v-btn>
        </div>

        <div v-else class="options-list">
          <div
            v-for="option in uciOptions"
            :key="option.name"
            class="option-item"
          >
            <!-- Conditional rendering for controls -->
            <div>
              <!-- 数值类型选项 (spin) -->
              <div v-if="option.type === 'spin'" class="option-row spin-option">
                <div class="option-header">
                  <label class="option-label">{{ option.name }}</label>
                  <span class="option-range"
                    >{{ $t('uciOptions.range') }}: {{ option.min }} -
                    {{ option.max }}</span
                  >
                </div>
                <div class="option-controls">
                  <v-text-field
                    v-model.number="option.currentValue"
                    :min="option.min"
                    :max="option.max"
                    type="number"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="number-input"
                    @update:model-value="updateOption(option.name, $event)"
                  ></v-text-field>
                </div>
              </div>

              <!-- 布尔类型选项 (check) -->
              <div
                v-else-if="option.type === 'check'"
                class="option-row check-option"
              >
                <div class="d-flex justify-space-between align-center">
                  <label class="option-label">{{ option.name }}</label>
                  <v-switch
                    v-model="option.currentValue as boolean"
                    color="primary"
                    class="option-switch flex-grow-0"
                    hide-details
                    @update:model-value="
                      updateOption(option.name, $event ?? false)
                    "
                  ></v-switch>
                </div>
              </div>

              <!-- 下拉选择类型选项 (combo) -->
              <div
                v-else-if="option.type === 'combo'"
                class="option-row combo-option"
              >
                <label class="option-label">{{ option.name }}</label>
                <v-select
                  v-model="option.currentValue as string"
                  :items="option.vars"
                  variant="outlined"
                  density="compact"
                  hide-details
                  class="option-select"
                  @update:model-value="updateOption(option.name, $event || '')"
                ></v-select>
              </div>

              <!-- 字符串类型选项 (string) -->
              <div
                v-else-if="option.type === 'string'"
                class="option-row string-option"
              >
                <label class="option-label">{{ option.name }}</label>
                <v-text-field
                  v-model="option.currentValue as string"
                  variant="outlined"
                  density="compact"
                  hide-details
                  class="option-input"
                  @update:model-value="updateOption(option.name, $event || '')"
                ></v-text-field>
              </div>

              <!-- 按钮类型选项 (button) -->
              <div
                v-else-if="option.type === 'button'"
                class="option-row button-option"
              >
                <div class="d-flex justify-space-between align-center">
                  <label class="option-label">{{ option.name }}</label>
                  <v-btn
                    color="primary"
                    variant="outlined"
                    class="execute-btn"
                    @click="executeButtonOption(option.name)"
                  >
                    {{ $t('uciOptions.execute') }}
                  </v-btn>
                </div>
              </div>
            </div>

            <div
              v-if="getOptionDescription(option.name)"
              class="option-description"
            >
              <v-icon size="small" class="description-icon"
                >mdi-information-outline</v-icon
              >
              <span class="description-text">{{
                getOptionDescription(option.name)
              }}</span>
            </div>
          </div>
        </div>
      </v-card-text>

      <v-card-actions class="dialog-actions">
        <div class="actions-container">
          <div class="action-buttons">
            <v-btn
              color="grey"
              @click="resetToDefaults"
              size="small"
              class="action-btn"
            >
              {{ $t('uciOptions.resetToDefaults') }}
            </v-btn>
            <v-btn
              color="primary"
              @click="refreshOptions"
              size="small"
              class="action-btn"
            >
              {{ $t('uciOptions.refreshOptions') }}
            </v-btn>
            <v-btn
              color="grey"
              @click="clearSettings"
              size="small"
              class="action-btn"
            >
              {{ $t('uciOptions.clearSettings') }}
            </v-btn>
          </div>
          <v-btn
            color="grey"
            @click="closeDialog"
            size="small"
            class="close-action-btn"
          >
            {{ $t('common.close') }}
          </v-btn>
        </div>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
  import { ref, computed, onMounted, watch, inject } from 'vue'
  import { useI18n } from 'vue-i18n'
  import { useConfigManager } from '../composables/useConfigManager'
  import { useTheme } from 'vuetify'

  // UCI option interface definition
  interface UciOption {
    name: string
    type: 'spin' | 'check' | 'combo' | 'string' | 'button'
    defaultValue: string | number | boolean
    currentValue: string | number | boolean
    min?: number
    max?: number
    vars?: string[]
  }

  // Component properties definition
  interface Props {
    modelValue: boolean
    engineId?: string
  }

  const props = withDefaults(defineProps<Props>(), {
    engineId: 'default',
  })

  // Component events definition
  const emit = defineEmits<{
    'update:modelValue': [value: boolean]
  }>()

  // Inject engine state
  const { t } = useI18n()
  const engineState = inject('engine-state') as any
  const { isEngineLoaded, engineOutput, uciOptionsText, currentEnginePath } =
    engineState

  // Theme detection
  const theme = useTheme()
  const isDark = computed(() => theme.global.current.value.dark)

  // Configuration manager
  const configManager = useConfigManager()

  // Reactive data
  const isLoading = ref(false)
  const uciOptions = ref<UciOption[]>([])
  const isWaitingForOptions = ref(false)
  const originalOptions = ref<Record<string, string | number | boolean>>({})

  // Detect if the device is mobile
  const isMobile = computed(() => {
    return window.innerWidth <= 768
  })

  // Computed property - dialog visibility state
  const isVisible = computed({
    get: () => props.modelValue,
    set: (value: boolean) => emit('update:modelValue', value),
  })

  // Engine path hash for config storage
  const enginePathHash = computed(() => {
    // THIS IS THE KEY CHANGE. Use the engine's unique ID.
    if (!props.engineId) return 'default'
    return props.engineId
  })

  // Send UCI command to the engine
  const sendUciCommand = (command: string) => {
    if (!isEngineLoaded.value) {
      return
    }

    // Directly call the engine's send method
    if (engineState.send) {
      engineState.send(command)
    }
  }

  // Function to parse UCI options text
  const parseUciOptions = (uciText: string): UciOption[] => {
    const options: UciOption[] = []
    const lines = uciText.split('\n')

    lines.forEach(line => {
      const trimmedLine = line.trim()
      if (trimmedLine.startsWith('option name ')) {
        const option = parseUciOptionLine(trimmedLine)
        if (option) {
          options.push(option)
        }
      }
    })

    return options
  }

  // Function to parse a single UCI option line
  const parseUciOptionLine = (line: string): UciOption | null => {
    const nameMatch = line.match(/option name (.+?) type (.+)/)
    if (!nameMatch) return null

    const name = nameMatch[1]
    const typeMatch = line.match(/type (\w+)/)
    if (!typeMatch) return null

    const type = typeMatch[1] as UciOption['type']

    // Create a basic option object
    const option: UciOption = {
      name,
      type,
      defaultValue: '',
      currentValue: '',
    }

    // Parse specific parameters based on type
    switch (type) {
      case 'spin':
        const defaultSpinMatch = line.match(/default (\d+)/)
        const minMatch = line.match(/min (\d+)/)
        const maxMatch = line.match(/max (\d+)/)

        option.defaultValue = defaultSpinMatch
          ? parseInt(defaultSpinMatch[1])
          : 0
        option.min = minMatch ? parseInt(minMatch[1]) : 0
        option.max = maxMatch ? parseInt(maxMatch[1]) : 100
        break

      case 'check':
        const defaultCheckMatch = line.match(/default (true|false)/)
        option.defaultValue = defaultCheckMatch
          ? defaultCheckMatch[1] === 'true'
          : false
        break

      case 'combo':
        const defaultComboMatch = line.match(/default (\w+)/)
        const varsMatch = line.match(/var (.+)/)

        option.defaultValue = defaultComboMatch ? defaultComboMatch[1] : ''
        option.vars = varsMatch ? varsMatch[1].split(' var ') : []
        break

      case 'string':
        const defaultStringMatch = line.match(/default (.+?)(?:\s|$)/)
        option.defaultValue = defaultStringMatch ? defaultStringMatch[1] : ''
        break

      case 'button':
        option.defaultValue = ''
        break
    }

    // Set current value to the default value
    option.currentValue = option.defaultValue

    return option
  }

  // Load saved option values from config file
  const loadSavedOptions = () => {
    const savedOptions = configManager.getUciOptions(enginePathHash.value)

    // Apply the saved values to the current options
    uciOptions.value.forEach(option => {
      if (savedOptions[option.name] !== undefined) {
        option.currentValue = savedOptions[option.name]
        // Immediately send the set command to the engine
        sendOptionToEngine(option.name, option.currentValue)
      }
    })
  }

  // Save option values to config file
  const saveOptionsToStorage = async () => {
    const optionsToSave: Record<string, string | number | boolean> = {}

    uciOptions.value.forEach(option => {
      if (option.type !== 'button') {
        optionsToSave[option.name] = option.currentValue
      }
    })

    await configManager.updateUciOptions(enginePathHash.value, optionsToSave)
  }

  // Send option set command to the engine
  const sendOptionToEngine = (
    name: string,
    value: string | number | boolean
  ) => {
    const command = `setoption name ${name} value ${value}`
    sendUciCommand(command)
  }

  // Function to update an option's value
  const updateOption = (name: string, value: string | number | boolean) => {
    const option = uciOptions.value.find(opt => opt.name === name)
    if (option) {
      option.currentValue = value
    }
  }

  // Function to execute a button-type option
  const executeButtonOption = (name: string) => {
    const command = `setoption name ${name}`
    sendUciCommand(command)
  }

  // Function to reset to default values
  const resetToDefaults = async () => {
    uciOptions.value.forEach(option => {
      option.currentValue = option.defaultValue
      // Send reset command to the engine
      if (option.type !== 'button') {
        sendOptionToEngine(option.name, option.defaultValue)
      }
    })

    // Clear config storage
    await configManager.clearUciOptions(enginePathHash.value)
  }

  // Function to refresh UCI options
  const refreshOptions = () => {
    if (!isEngineLoaded.value) {
      alert(t('uciOptions.noEngineLoaded'))
      return
    }

    isLoading.value = true

    // If options are not in cache, actively request them
    if (!uciOptionsText.value || uciOptionsText.value.trim() === '') {
      sendUciCommand('uci')
    }

    // Parse cached options
    setTimeout(() => {
      isLoading.value = false
      const options = parseUciOptions(uciOptionsText.value)
      uciOptions.value = options
      loadSavedOptions()

      // Store original values after loading
      originalOptions.value = {}
      uciOptions.value.forEach(option => {
        if (option.type !== 'button') {
          originalOptions.value[option.name] = option.currentValue
        }
      })
    }, 100)
  }

  // Function to close the dialog
  const closeDialog = async () => {
    try {
      // Identify modified options and send `setoption` command
      for (const option of uciOptions.value) {
        if (
          option.type !== 'button' &&
          originalOptions.value[option.name] !== option.currentValue
        ) {
          sendOptionToEngine(option.name, option.currentValue)
        }
      }

      // Save all current option values to config. This is an async operation that might fail.
      await saveOptionsToStorage()
    } catch (error) {
      // If saving fails, log the error for debugging purposes.
      // This prevents an unhandled promise rejection from stopping execution.
      console.error('Failed to save UCI options on close:', error)
    } finally {
      // CRUCIAL: Ensure the dialog is always closed, regardless of whether the save operation succeeded or failed.
      // This makes the close button robust and reliable for the user.
      isVisible.value = false
    }
  }

  // Watch engine output to get UCI options
  const parseEngineOutput = () => {
    if (!isWaitingForOptions.value) return

    // Collect all engine output
    const allOutput = (engineOutput.value as { kind: string; text: string }[])
      .filter(line => line.kind === 'recv')
      .map(line => line.text)
      .join('\n')

    // Check if 'uciok' is received (indicates options sending is complete)
    if (allOutput.includes('uciok')) {
      isWaitingForOptions.value = false
      isLoading.value = false

      // Parse options
      const options = parseUciOptions(allOutput)
      uciOptions.value = options

      // Load saved option values
      loadSavedOptions()
    }
  }

  // Watch for changes in engine output
  watch(() => engineOutput.value, parseEngineOutput, { deep: true })

  // Watch for changes in engine load status
  watch(
    () => isEngineLoaded.value,
    newVal => {
      if (newVal && currentEnginePath.value) {
        // After the engine is loaded, refresh options with a delay to ensure the engine is ready
        setTimeout(() => {
          refreshOptions()
        }, 500)
      }
    }
  )

  // Watch the dialog's open state
  watch(isVisible, newVal => {
    if (newVal && isEngineLoaded.value) {
      // Automatically refresh options when the dialog opens
      refreshOptions()
    }
  })

  // Initialization after component is mounted
  onMounted(async () => {
    // Load configuration first
    await configManager.loadConfig()

    // If the engine is already loaded, get options immediately
    if (isEngineLoaded.value) {
      setTimeout(() => {
        refreshOptions()
      }, 500)
    }
  })

  // Function to get option description from i18n
  const getOptionDescription = (optionName: string): string => {
    // Construct the full key path to the specific translation.
    // For example: 'uciOptions.optionDescriptions.Threads' or 'uciOptions.optionDescriptions.Debug Log File'
    const fullKey = `uciOptions.optionDescriptions.${optionName}`

    // Use the t() function to get the final translated string directly.
    const description = t(fullKey)

    // IMPORTANT: If the t() function cannot find a corresponding translation, it will return the key itself.
    // We need to check if the returned value is equal to the key we provided to determine if the translation exists.
    // If they are equal, it means the translation was not found, so we return an empty string, which will cause the v-if to hide the element.
    return description !== fullKey ? description : ''
  }

  // Function to clear settings
  const clearSettings = async () => {
    if (confirm(t('uciOptions.confirmClearSettings'))) {
      // Clear config storage
      await configManager.clearUciOptions(enginePathHash.value)

      // Reset all options to their default values
      uciOptions.value.forEach(option => {
        option.currentValue = option.defaultValue
        // Send reset command to the engine
        if (option.type !== 'button') {
          sendOptionToEngine(option.name, option.defaultValue)
        }
      })

      // console.log(t('uciOptions.settingsCleared'));
    }
  }

  // Expose methods to the parent component
  defineExpose({
    refreshOptions,
    resetToDefaults,
    clearSettings,
  })
</script>

<style lang="scss" scoped>
  .uci-options-card {
    border-radius: 12px;
    overflow: hidden;
  }

  .dialog-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 16px 20px;
    position: relative;

    .title-text {
      font-size: 18px;
      font-weight: 600;
    }

    .close-btn {
      color: white;
      margin-right: -8px;
    }
  }

  .options-container {
    max-height: 70vh;
    overflow-y: auto;
    padding: 16px;
  }

  .loading-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    padding: 40px 20px;

    .loading-text {
      color: rgb(var(--v-theme-on-surface));
      font-size: 16px;
      text-align: center;
    }
  }

  .empty-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    padding: 40px 20px;
    color: rgb(var(--v-theme-on-surface));

    .empty-text {
      margin: 0;
      font-size: 16px;
      text-align: center;
      line-height: 1.5;
      color: rgb(var(--v-theme-on-surface));
    }

    .empty-text-secondary {
      margin: 0;
      font-size: 14px;
      text-align: center;
      line-height: 1.4;
      color: rgb(var(--v-theme-on-surface));
      opacity: 0.7;
    }

    .action-btn {
      min-width: 120px;
    }
  }

  .options-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .option-item {
    border: 1px solid rgb(var(--v-border-color));
    border-radius: 12px;
    padding: 16px;
    background: rgb(var(--v-theme-surface));
    transition: all 0.2s ease;

    &:hover {
      background: rgb(var(--v-theme-surface-variant));
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
  }

  .v-theme--dark .option-item:hover {
    background: rgb(30, 30, 30) !important;
  }

  .v-theme--light .option-item:hover {
    background: rgb(245, 245, 245) !important;
  }

  .v-theme--dark .dialog-actions {
    background: rgb(30, 30, 30) !important;
  }

  .v-theme--light .dialog-actions {
    background: rgb(245, 245, 245) !important;
  }

  .option-row {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 48px;
  }

  .option-header {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .option-label {
    font-weight: 600;
    color: rgb(var(--v-theme-on-surface));
    font-size: 16px;
    line-height: 1.4;
  }

  .option-range {
    font-size: 12px;
    color: rgb(var(--v-theme-on-surface));
    font-weight: 400;
    opacity: 0.7;
  }

  .option-controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 100%;
  }

  .number-input {
    max-width: 120px;
    align-self: flex-start;
  }

  .option-switch {
    align-self: flex-end;
    margin-top: 8px;
  }

  .option-select,
  .option-input {
    width: 100%;
  }

  .execute-btn {
    align-self: flex-end;
    margin-top: 8px;
    min-width: 100px;
  }

  .option-description {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 12px;
    padding: 10px 12px;
    background-color: rgba(var(--v-theme-primary), 0.08);
    border-radius: 8px;
    color: rgb(var(--v-theme-on-surface));

    .description-icon {
      color: inherit;
      margin-top: 2px;
    }

    .description-text {
      flex: 1;
    }
  }

  // Tweak for dark mode
  .v-theme--dark .option-description {
    background-color: rgba(var(--v-theme-primary), 0.15);
  }

  .dialog-actions {
    padding: 16px 20px;
    background: rgb(var(--v-theme-surface));

    .actions-container {
      display: flex;
      flex-direction: column;
      gap: 12px;
      width: 100%;
    }

    .action-buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: center;
    }

    .action-btn {
      text-transform: none;
      font-weight: 500;
      font-size: 14px;
      min-width: 80px;
    }

    .close-action-btn {
      text-transform: none;
      font-weight: 500;
      font-size: 14px;
      align-self: center;
      min-width: 100px;
    }
  }

  @media (max-width: 768px) {
    .options-container {
      max-height: 60vh;
      padding: 12px;
    }

    .option-item {
      padding: 12px;
      border-radius: 8px;
    }

    .option-label {
      font-size: 15px;
    }

    .dialog-actions {
      padding: 12px 16px;

      .action-buttons {
        gap: 6px;
      }

      .action-btn {
        font-size: 13px;
        min-width: 70px;
        padding: 8px 12px;
      }

      .close-action-btn {
        font-size: 13px;
        min-width: 80px;
        padding: 8px 16px;
      }
    }

    .dialog-title {
      padding: 12px 16px;

      .title-text {
        font-size: 16px;
      }
    }

    .loading-section,
    .empty-section {
      padding: 30px 16px;
    }

    .loading-text,
    .empty-text {
      font-size: 14px;
    }
  }

  // Extra small screen optimization
  @media (max-width: 480px) {
    .options-container {
      max-height: 55vh;
      padding: 8px;
    }

    .option-item {
      padding: 10px;
    }

    .option-label {
      font-size: 14px;
    }

    .dialog-actions {
      padding: 10px 12px;

      .action-btn {
        font-size: 12px;
        min-width: 60px;
        padding: 6px 10px;
      }

      .close-action-btn {
        font-size: 12px;
        min-width: 70px;
        padding: 6px 12px;
      }
    }

    .dialog-title {
      padding: 10px 12px;

      .title-text {
        font-size: 15px;
      }
    }
  }

  // Scrollbar styles
  .options-container::-webkit-scrollbar {
    width: 4px;
  }

  .options-container::-webkit-scrollbar-track {
    background: rgb(var(--v-theme-surface-variant));
    border-radius: 2px;
  }

  .options-container::-webkit-scrollbar-thumb {
    background: rgb(var(--v-theme-outline));
    border-radius: 2px;

    &:hover {
      background: rgb(var(--v-theme-outline-variant));
    }
  }
</style>
