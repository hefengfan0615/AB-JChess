<template>
  <div v-if="visible" class="dialog-mask">
    <div class="dialog-container">
      <div class="dialog-title">{{ t('gameConfirm.clearHistoryTitle') }}</div>
      <div class="dialog-message">
        {{ t('gameConfirm.clearHistoryMessage') }}
      </div>
      <div class="dialog-actions">
        <button class="btn confirm" @click="onConfirm">
          {{ t('gameConfirm.confirm') }}
        </button>
        <button class="btn cancel" @click="onCancel">
          {{ t('gameConfirm.cancel') }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
  import { useI18n } from 'vue-i18n'
  const props = defineProps({
    visible: Boolean,
    onConfirm: Function,
    onCancel: Function,
  })
  const { t } = useI18n()
</script>

<style lang="scss" scoped>
  .dialog-mask {
    position: fixed;
    z-index: 2000;
    left: 0;
    top: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .dialog-container {
    background: rgb(var(--v-theme-surface));
    color: rgb(var(--v-theme-on-surface));
    border-radius: 8px;
    min-width: 320px;
    max-width: 90vw;
    box-shadow: 0 2px 16px rgba(0, 0, 0, 0.18);
    padding: 24px 20px 16px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .dialog-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 12px;
    color: rgb(var(--v-theme-on-surface));
  }
  .dialog-message {
    font-size: 15px;
    margin-bottom: 20px;
    text-align: center;
    color: rgb(var(--v-theme-on-surface));
  }
  .dialog-actions {
    display: flex;
    gap: 16px;
  }
  .btn {
    min-width: 72px;
    padding: 6px 18px;
    border: none;
    border-radius: 4px;
    font-size: 15px;
    cursor: pointer;
    transition: background 0.2s;
  }
  .btn.confirm {
    background: rgb(var(--v-theme-primary));
    color: rgb(var(--v-theme-on-primary));
  }
  .btn.cancel {
    background: rgb(var(--v-theme-surface-variant));
    color: rgb(var(--v-theme-on-surface-variant));
  }
  .btn.confirm:hover {
    background: rgb(var(--v-theme-primary-darken-1));
  }
  .btn.cancel:hover {
    background: rgb(var(--v-theme-surface-variant-darken-1));
  }
</style>
