import { Block } from 'baseui/block'
import React from 'react'
import { Client as Styletron } from 'styletron-engine-monolithic'
import { Provider as StyletronProvider } from 'styletron-react'
import { BaseProvider, createLightTheme, createDarkTheme } from 'baseui'
import { SnackbarProvider, PLACEMENT } from 'baseui/snackbar'
import './layout.css'
import { colors } from 'baseui/tokens'
import { Header } from './header'
import { MAX_CONTENT_WIDTH } from '../../config/theme'

const engine = new Styletron()

// A slightly warmer, lower-contrast light palette. Base Web's default is very
// stark white-on-white, which makes every card edge shout.
const lightTheme = createLightTheme({
  colors: {
    linkVisited: colors.black,
    backgroundPrimary: '#FFFFFF',
    backgroundSecondary: '#F7F7F8',
    backgroundTertiary: '#EFEFF1',
    contentPrimary: '#1F2328',
    contentSecondary: '#5B6169',
    contentTertiary: '#848B94',
    borderOpaque: '#E4E4E7',
    borderTransparent: 'rgba(31, 35, 40, 0.08)',
  },
})

const darkTheme = createDarkTheme({
  colors: {
    primaryA: '#F8FAFC',
    primaryB: '#171E27',
    primary: '#8DB8FF',
    accent: '#8DB8FF',
    backgroundPrimary: '#171E27',
    backgroundSecondary: '#202A36',
    backgroundTertiary: '#2A3543',
    backgroundStateDisabled: '#26313E',
    backgroundAccent: '#8DB8FF',
    backgroundAccentLight: '#253B59',
    contentPrimary: '#F8FAFC',
    contentSecondary: '#DCE3EC',
    contentTertiary: '#B5C0CE',
    contentStateDisabled: '#8995A5',
    contentAccent: '#A9CBFF',
    contentOnColor: '#0D141C',
    borderOpaque: '#465568',
    borderTransparent: 'rgba(248, 250, 252, 0.18)',
    borderSelected: '#A9CBFF',
    borderAccent: '#8DB8FF',
    linkText: '#A9CBFF',
    linkVisited: '#A9CBFF',
    inputFill: '#202A36',
    inputFillActive: '#263342',
    inputFillError: '#202A36',
    inputFillPositive: '#202A36',
    inputBorder: '#56667A',
    inputBorderActive: '#A9CBFF',
    inputPlaceholder: '#B5C0CE',
    buttonPrimaryFill: '#8DB8FF',
    buttonPrimaryText: '#0D141C',
    buttonPrimaryHover: '#A2C6FF',
    buttonPrimaryActive: '#78A5E8',
    buttonSecondaryFill: '#2A3543',
    buttonSecondaryText: '#F8FAFC',
    buttonSecondaryHover: '#344152',
    buttonTertiaryText: '#E7ECF3',
    buttonTertiaryHover: '#2A3543',
    tickFill: '#202A36',
    tickFillSelected: '#8DB8FF',
    tickMarkFill: '#0D141C',
    toggleFill: '#202A36',
    toggleFillChecked: '#8DB8FF',
    toggleTrackFill: '#56667A',
    toggleTrackFillChecked: '#4B70A5',
    fileUploaderBackgroundColor: '#202A36',
    fileUploaderBackgroundColorActive: '#253B59',
    fileUploaderBorderColorActive: '#8DB8FF',
    fileUploaderMessageColor: '#F8FAFC',
    fileUploaderErrorColor: '#F28B82',
    snackbarBackground: '#2A3543',
    snackbarText: '#F8FAFC',
  },
})

const DARK_BACKGROUND = '#0D1218'
const LIGHT_BACKGROUND = '#FAFAFA'
const THEME_STORAGE_KEY = 'ai100-gpt-theme'

type LayoutProps = {
  children: React.ReactNode
}

export function Layout(props: LayoutProps) {
  const { children } = props

  const [isDark, setIsDark] = React.useState(() => {
    if (typeof window !== 'undefined') {
      try {
        const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
        if (savedTheme === 'dark' || savedTheme === 'light') {
          return savedTheme === 'dark'
        }
      } catch {
        // Some privacy modes disable storage; the system preference still works.
      }
      return window.matchMedia('(prefers-color-scheme: dark)').matches
    }
    return false
  })

  React.useEffect(() => {
    document.body.style.backgroundColor = isDark ? DARK_BACKGROUND : LIGHT_BACKGROUND
    document.body.style.margin = '0'
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, isDark ? 'dark' : 'light')
    } catch {
      // Theme selection remains usable for this visit when storage is unavailable.
    }
  }, [isDark])

  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={isDark ? darkTheme : lightTheme}>
        <SnackbarProvider placement={PLACEMENT.bottom}>
          <Block
            display="flex"
            flexDirection="column"
            backgroundColor={isDark ? DARK_BACKGROUND : LIGHT_BACKGROUND}
            minHeight="100vh"
          >
            {/* The bar spans the window; only the reading column is narrow. */}
            <Header isDark={isDark} onToggleDark={() => setIsDark((d) => !d)} />
            <Block
              display="flex"
              flexDirection="column"
              width="100%"
              maxWidth={MAX_CONTENT_WIDTH}
              alignSelf="center"
              paddingTop="scale800"
            >
              {children}
            </Block>
          </Block>
        </SnackbarProvider>
      </BaseProvider>
    </StyletronProvider>
  )
}
