import { Block } from 'baseui/block'
import {
  HeaderNavigation,
  ALIGN,
  StyledNavigationList,
  StyledNavigationItem,
} from 'baseui/header-navigation'
import { StyledLink } from 'baseui/link'
import { GoHubot } from 'react-icons/go'
import { IoMoon, IoSunny } from 'react-icons/io5'
import { Button, SIZE, KIND } from 'baseui/button'

import { BASE_PATH } from '../../config/links'
import { WINDOW_PADDING_HORIZONTAL } from '../../config/theme'

type HeaderProps = {
  isDark?: boolean
  onToggleDark?: () => void
}

export function Header(props: HeaderProps) {
  const { isDark = false, onToggleDark = () => {} } = props

  return (
    <Block marginBottom="scale800">
      <HeaderNavigation overrides={{ Root: { style: { borderBottomWidth: '2px' } } }}>
        <StyledNavigationList $align={ALIGN.left}>
          <StyledNavigationItem $style={{ paddingLeft: WINDOW_PADDING_HORIZONTAL }}>
            <Block
              display="flex"
              flexDirection="row"
              alignItems="center"
              justifyContent="center"
              color="contentAccent"
              $style={{ lineHeight: 0 }}
            >
              <Block marginRight="5px">
                <GoHubot size="20" />
              </Block>
              <StyledLink
                href={BASE_PATH}
                $style={{
                  color: 'inherit',
                  textDecoration: 'none',
                  fontSize: '18px',
                  fontWeight: 700,
                  letterSpacing: '0.02em',
                }}
              >
                AI100-GPT
              </StyledLink>
            </Block>
          </StyledNavigationItem>
        </StyledNavigationList>

        <StyledNavigationList $align={ALIGN.right}>
          <StyledNavigationItem $style={{ paddingRight: WINDOW_PADDING_HORIZONTAL }}>
            <Button onClick={onToggleDark} size={SIZE.compact} kind={KIND.tertiary}>
              {isDark ? <IoSunny size={18} /> : <IoMoon size={18} />}
            </Button>
          </StyledNavigationItem>
        </StyledNavigationList>
      </HeaderNavigation>
    </Block>
  )
}
