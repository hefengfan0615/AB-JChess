<template>
  <v-dialog
    v-model="isVisible"
    :max-width="isMobile ? '95vw' : '800px'"
    persistent
  >
    <v-card class="jai-options-card">
      <v-card-title class="dialog-title">
        <span class="title-text">{{ $t('jaiOptions.title') }}</span>
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
          <span class="loading-text">{{ $t('jaiOptions.loadingText') }}</span>
        </div>

        <div v-else-if="!isEngineLoaded" class="empty-section">
          <v-icon size="64" color="grey">mdi-engine-off</v-icon>
          <p class="empty-text">{{ $t('jaiOptions.noEngineLoaded') }}</p>
          <p class="empty-text-secondary">
            {{ $t('jaiOptions.pleaseLoadEngineFirst') }}
          </p>
        </div>

        <div v-else-if="jaiOptions.length === 0" class="empty-section">
          <v-icon size="64" color="grey">mdi-cog-off</v-icon>
          <p class="empty-text">{{ $t('jaiOptions.noOptionsAvailable') }}</p>
          <v-btn
            color="primary"
            @click="refreshOptions"
            size="large"
            class="action-btn"
          >
            {{ $t('jaiOptions.refreshOptions') }}
          </v-btn>
        </div>

        <div v-else class="options-list">
          <div
            v-for="option in jaiOptions"
            :key="option.name"
            class="option-item"
          >
            <!-- Conditional rendering for controls -->
            <div>
              <!-- Numeric type option (spin) -->
              <div v-if="option.type === 'spin'" class="option-row spin-option">
                <div class="option-header">
                  <label class="option-label">{{ option.name }}</label>
                  <span class="option-range"
                    >{{ $t('jaiOptions.range') }}: {{ option.min }} -
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

              <!-- Boolean type option (check) -->
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

              <!-- Dropdown select type option (combo) -->
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

              <!-- String type option (string) -->
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

              <!-- Button type option (button) -->
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
                    {{ $t('jaiOptions.execute') }}
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
              {{ $t('jaiOptions.resetToDefaults') }}
            </v-btn>
            <v-btn
              color="primary"
              @click="refreshOptions"
              size="small"
              class="action-btn"
            >
              {{ $t('jaiOptions.refreshOptions') }}
            </v-btn>
            <v-btn
              color="grey"
              @click="clearSettings"
              size="small"
              class="action-btn"
            >
              {{ $t('jaiOptions.clearSettings') }}
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
  import type { JaiOption } from '../composables/useJaiEngine'

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

  // Inject JAI engine state
  const { t } = useI18n()
  const jaiEngineState = inject('jai-engine-state') as any
  const { isEngineLoaded, jaiOptionsText, jaiOptions, setJaiOption, send } =
    jaiEngineState

  // Theme detection
  const theme = useTheme()
  const isDark = computed(() => theme.global.current.value.dark)

  // Configuration manager
  const configManager = useConfigManager()

  // Reactive data
  const isLoading = ref(false)
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
    // Use the engine's unique ID for JAI options storage
    if (!props.engineId) return 'default'
    return props.engineId
  })

  // Load saved option values from config file
  const loadSavedOptions = () => {
    const savedOptions = configManager.getJaiOptions(enginePathHash.value)

    // Apply the saved values to the current options
    jaiOptions.value.forEach((option: JaiOption) => {
      if (savedOptions[option.name] !== undefined) {
        option.currentValue = savedOptions[option.name]
        // Immediately send the set command to the engine
        setJaiOption(option.name, option.currentValue)
      }
    })
  }

  // Save option values to config file
  const saveOptionsToStorage = async () => {
    const optionsToSave: Record<string, string | number | boolean> = {}

    jaiOptions.value.forEach((option: JaiOption) => {
      if (option.type !== 'button') {
        optionsToSave[option.name] = option.currentValue
      }
    })

    await configManager.updateJaiOptions(enginePathHash.value, optionsToSave)
  }

  // Function to update an option's value
  const updateOption = (name: string, value: string | number | boolean) => {
    const option = jaiOptions.value.find((opt: JaiOption) => opt.name === name)
    if (option) {
      option.currentValue = value
    }
  }

  // Function to execute a button-type option
  const executeButtonOption = (name: string) => {
    const command = `setoption name ${name}`
    send(command)
  }

  // Function to reset to default values
  const resetToDefaults = async () => {
    jaiOptions.value.forEach((option: JaiOption) => {
      option.currentValue = option.defaultValue
      // Send reset command to the engine
      if (option.type !== 'button') {
        setJaiOption(option.name, option.defaultValue)
      }
    })

    // Clear config storage
    await configManager.clearJaiOptions(enginePathHash.value)
  }

  // Function to refresh JAI options
  const refreshOptions = () => {
    if (!isEngineLoaded.value) {
      alert(t('jaiOptions.noEngineLoaded'))
      return
    }

    isLoading.value = true

    // If options are not in cache, actively request them
    if (!jaiOptionsText.value || jaiOptionsText.value.trim() === '') {
      send('jai')
    }

    // Load saved options after a short delay
    setTimeout(() => {
      isLoading.value = false
      loadSavedOptions()

      // Store original values after loading
      originalOptions.value = {}
      jaiOptions.value.forEach((option: JaiOption) => {
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
      for (const option of jaiOptions.value) {
        if (
          option.type !== 'button' &&
          originalOptions.value[option.name] !== option.currentValue
        ) {
          setJaiOption(option.name, option.currentValue)
        }
      }

      // Save all current option values to config
      await saveOptionsToStorage()
    } catch (error) {
      console.error('Failed to save JAI options on close:', error)
    } finally {
      isVisible.value = false
    }
  }

  // Watch for changes in engine load status
  watch(
    () => isEngineLoaded.value,
    newVal => {
      if (newVal) {
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
    const fullKey = `jaiOptions.optionDescriptions.${optionName}`

    // Use the t() function to get the final translated string directly.
    const description = t(fullKey)

    // If the translation was not found, return empty string
    return description !== fullKey ? description : ''
  }

  // Function to clear settings
  const clearSettings = async () => {
    if (confirm(t('jaiOptions.confirmClearSettings'))) {
      // Clear config storage
      await configManager.clearJaiOptions(enginePathHash.value)

      // Reset all options to their default values
      jaiOptions.value.forEach((option: JaiOption) => {
        option.currentValue = option.defaultValue
        // Send reset command to the engine
        if (option.type !== 'button') {
          setJaiOption(option.name, option.defaultValue)
        }
      })
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
  .jai-options-card {
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
