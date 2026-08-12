import React from 'react'
import { Block } from 'baseui/block'
import { FormControl } from 'baseui/form-control'
import { Button, KIND } from 'baseui/button'
import { RiUpload2Line } from 'react-icons/ri'

type StyleFileProps = {
  onUpload: (file: File) => void
  baseLabel: string
  disabled?: boolean
}

/**
 * Loads a style someone else saved. The file holds only the adapter, so it is a
 * small fraction of the model and only fits the base it was trained on.
 */
export function StyleFile(props: StyleFileProps) {
  const { onUpload, baseLabel, disabled = false } = props
  const inputRef = React.useRef<HTMLInputElement>(null)

  return (
    <Block>
      <FormControl
        label="Load a style"
        caption={`Open a style file saved from this page. It has to be one trained on the ${baseLabel} model.`}
      >
        {/*
          One element, not two: FormControl passes its child through
          React.Children.only, so the button and the hidden file input have to
          be wrapped or the whole page fails to render.
        */}
        <Block>
          <Button
            kind={KIND.secondary}
            disabled={disabled}
            onClick={() => inputRef.current?.click()}
            startEnhancer={() => <RiUpload2Line />}
            overrides={{ Root: { style: { width: '100%' } } }}
          >
            Choose a style file
          </Button>
          <input
            ref={inputRef}
            type="file"
            accept="application/json,.json"
            disabled={disabled}
            style={{ display: 'none' }}
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) onUpload(file)
              // Let selecting the same file twice trigger another change event.
              event.target.value = ''
            }}
          />
        </Block>
      </FormControl>
    </Block>
  )
}
