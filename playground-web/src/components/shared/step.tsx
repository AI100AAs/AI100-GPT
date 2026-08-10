import { Block } from 'baseui/block'
import { useStyletron } from 'baseui'
import { Accordion, Panel } from 'baseui/accordion'

type StepProps = {
  title: string
  children: React.ReactNode
  accordion?: boolean
  closed?: boolean
}

export function Step(props: StepProps) {
  const { title, children, accordion = false, closed = false } = props
  const [, theme] = useStyletron()
  return (
    <Block marginBottom="scale600">
      <Block
        backgroundColor="backgroundPrimary"
        padding="scale750"
        $style={{
          borderRadius: '14px',
          border: `1px solid ${theme.colors.borderOpaque}`,
          // A single soft shadow reads as depth; Base Web's default card border
          // plus shadow reads as a box drawn around everything.
          boxShadow: '0 1px 2px rgba(16, 24, 40, 0.04)',
        }}
      >
        <Accordion
          accordion={accordion}
          initialState={{ expanded: closed ? [] : ['panel'] }}
          renderAll
          overrides={{
            Header: {
              style: {
                margin: 0,
                padding: 0,
                fontSize: '17px',
                fontWeight: 600,
                lineHeight: '24px',
                letterSpacing: '-0.01em',
              },
            },
            Content: {
              style: {
                margin: 0,
                paddingTop: '18px',
                paddingRight: 0,
                paddingLeft: 0,
                paddingBottom: 0,
                borderBottomWidth: 0,
                backgroundColor: 'transparent',
              },
            },
            PanelContainer: { style: { borderBottomWidth: 0 } },
          }}
        >
          <Panel key="panel" title={title}>
            {children}
          </Panel>
        </Accordion>
      </Block>
    </Block>
  )
}
