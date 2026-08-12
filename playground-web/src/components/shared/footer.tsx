import { Block } from 'baseui/block'
import { StyledLink } from 'baseui/link'

export function Footer() {
  return (
    <Block
      display="flex"
      marginBottom="scale1000"
      flexDirection="column"
      alignItems="center"
      color="contentTertiary"
      $style={{ fontSize: '12px', lineHeight: '18px', textAlign: 'center' }}
    >
      <Block
        marginTop="scale1000"
        paddingTop="scale600"
        paddingLeft="scale600"
        paddingRight="scale600"
        display="flex"
        flexDirection="column"
        gridGap="8px"
        justifyContent="center"
        alignItems="center"
        $style={{ borderTop: '2px solid rgba(127, 127, 127, 0.18)' }}
      >
        <Block>
          Developed for{' '}
          <StyledLink href="https://www.cs.ubc.ca/~kevinlb/teaching/ai100/">
            UBC's AI 100: Introduction to Artificial Intelligence
          </StyledLink>
          .
        </Block>
        <Block>
          Based on{' '}
          <StyledLink href="https://github.com/trekhleb/homemade-gpt-js">
            homemade-gpt-js
          </StyledLink>{' '}
          and Andrej Karpathy's{' '}
          <StyledLink href="https://github.com/karpathy/minGPT">minGPT</StyledLink>{' '}
          and{' '}
          <StyledLink href="https://www.youtube.com/watch?v=kCc8FmEb1nY">
            Let's build GPT
          </StyledLink>
          .
        </Block>
      </Block>
    </Block>
  )
}
