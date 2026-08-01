import { Block } from 'baseui/block'

export function Footer() {
  return (
    <Block display="flex" marginBottom="scale1000" flexDirection="column">
      <Block
        marginTop="scale1000"
        paddingTop="scale1000"
        display="flex"
        flexDirection="row"
        gridGap="5px"
        justifyContent="center"
        $style={{ borderTop: '2px solid rgba(127, 127, 127, 0.18)' }}
      />
    </Block>
  )
}
