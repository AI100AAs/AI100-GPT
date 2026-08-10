import { Block } from 'baseui/block'
import { FormControl } from 'baseui/form-control'
import { FileUploader } from 'baseui/file-uploader'

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
  return (
    <Block>
      <FormControl
        label="Load a style"
        caption={`Open a style file saved from this page. It has to be one trained on the ${baseLabel} model.`}
      >
        <FileUploader
          accept="application/json"
          multiple={false}
          disabled={disabled}
          onDrop={(accepted) => {
            if (accepted.length) onUpload(accepted[0])
          }}
        />
      </FormControl>
    </Block>
  )
}
