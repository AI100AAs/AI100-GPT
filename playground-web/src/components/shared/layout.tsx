import { Block } from 'baseui/block'
import React from 'react'
import { Client as Styletron } from 'styletron-engine-monolithic'
import { Provider as StyletronProvider } from 'styletron-react'
import { BaseProvider, createLightTheme, createDarkTheme } from 'baseui'
import { SnackbarProvider } from 'baseui/snackbar'
import './layout.css'
import { colors } from 'baseui/tokens'
import { Header } from './header'

const engine = new Styletron()

const lightTheme = createLightTheme({
  colors: {
    linkVisited: colors.black,
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

type LayoutProps = {
  children: React.ReactNode
}

export function Layout(props: LayoutProps) {
  const { children } = props

  const [isDark, setIsDark] = React.useState(() => {
    if (typeof window !== 'undefined') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches
    }
    return false
  })

  React.useEffect(() => {
    document.body.style.backgroundColor = isDark ? DARK_BACKGROUND : '#ffffff'
    document.body.style.margin = '0'
  }, [isDark])

  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={isDark ? darkTheme : lightTheme}>
        <SnackbarProvider>
          <Block
            display="flex"
            flexDirection="column"
            alignItems="center"
            backgroundColor={isDark ? DARK_BACKGROUND : '#ffffff'}
            minHeight="100vh"
          >
            <Block display="flex" flexDirection="column" width="100%" maxWidth="1200px">
              <Header isDark={isDark} onToggleDark={() => setIsDark((d) => !d)} />
              {children}
            </Block>
          </Block>
        </SnackbarProvider>
      </BaseProvider>
    </StyletronProvider>
  )
}
