import { createApp } from 'vue'
import App from './App.vue'

// Import Vuetify
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@mdi/font/css/materialdesignicons.css' // Import MDI icon styles

// Import i18n
import i18n from './i18n'
import { usePanelManager } from './composables/usePanelManager'

const vuetify = createVuetify({
  components,
  directives,
  icons: {
    defaultSet: 'mdi', // Set default icon set to mdi
  },
  // Add theme configuration for dark mode support
  theme: {
    defaultTheme: 'light',
    themes: {
      light: {
        dark: false,
        colors: {
          primary: '#1976D2',
          secondary: '#424242',
          accent: '#82B1FF',
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FF9800',
          background: '#F5F5F5',
          surface: '#FFFFFF',
          button: '#FFFFFF',
          'on-background': '#000000',
          'on-surface': '#000000',
        },
      },
      dark: {
        dark: true,
        colors: {
          primary: '#2196F3',
          secondary: '#424242',
          accent: '#FF4081',
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FF9800',
          background: '#202020',
          surface: '#121212',
          button: '#1A1A1A',
          'on-background': '#FFFFFF',
          'on-surface': '#FFFFFF',
        },
      },
    },
  },
})

const app = createApp(App)

// Initialize i18n
app.use(i18n)

// Initialize Vuetify
app.use(vuetify)

// Initialize panel manager
const panelManager = usePanelManager()
panelManager.initialize()

// Mount the app
app.mount('#app')
