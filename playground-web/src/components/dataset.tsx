import React from 'react'
import { Dataset as DatasetT, CharDataset } from '@gpt/model'
import { Block } from 'baseui/block'
import { SegmentedControl, Segment } from 'baseui/segmented-control'
import { BASE_DATASETS, BackendId, BaseDatasetId } from '../types/playground'
import { FaFeatherAlt } from 'react-icons/fa'
import { SIZE } from 'baseui/input'
import { StyledLink } from 'baseui/link'
import { Textarea } from 'baseui/textarea'
import { FormControl } from 'baseui/form-control'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { Skeleton } from 'baseui/skeleton'
import { BASE_PATH } from '../config/links'
import { Notification } from './shared/notification'
import { FadeIn } from './shared/fade'
import { FaUtensils } from 'react-icons/fa'
import { Count } from './shared/count'

const inputSize = SIZE.mini

type StepProps = {
  dataset: DatasetT | undefined
  backend: BackendId | undefined
  onChange?: (dataset: DatasetT, datasetId: BaseDatasetId) => Promise<void>
  showTechnicalDetails?: boolean
}

export function Dataset(props: StepProps) {
  const {
    onChange = () => {},
    dataset,
    backend,
    showTechnicalDetails = false,
  } = props

  const [datasetId, setDatasetId] = React.useState<BaseDatasetId>('shakespeare')
  const [isLoading, setIsLoading] = React.useState<boolean>()
  const [errorMessage, setErrorMessage] = React.useState<string>()

  const onDatasetInit = async () => {
    onDatasetChange(datasetId)
  }

  const onDatasetChange = async (nextDatasetId: BaseDatasetId) => {
    setIsLoading(true)
    setErrorMessage(undefined)
    try {
      const textSourceURL = `${BASE_PATH}/${BASE_DATASETS[nextDatasetId].file}`
      // The pretrained checkpoints use a token index shift of zero, so one of
      // their output classes belongs to the padding token and the alphabet is
      // one character shorter than the class count.
      const nextDataset = await CharDataset({ textSourceURL, reserveMaskClass: true })
      await onChange(nextDataset, nextDatasetId)
      setDatasetId(nextDatasetId)
    } catch (err) {
      setErrorMessage((err as Error).message)
    }
    setIsLoading(false)
  }

  React.useEffect(() => {
    if (!backend) return
    if (dataset === undefined) {
      onDatasetInit()
    }
  }, [backend, dataset])

  const error = errorMessage && (
    <Notification kind="negative">{errorMessage}</Notification>
  )

  const loader = isLoading && (
    <FadeIn>
      <Block marginTop="scale300">
        <Skeleton rows={3} height="220px" width="100%" animation autoSizeRows />
      </Block>
    </FadeIn>
  )

  const segments = (
    <Block paddingBottom="scale400">
      <FadeIn>
        <SegmentedControl
          activeKey={datasetId}
          disabled={isLoading}
          onChange={({ activeKey }) => {
            onDatasetChange(activeKey as BaseDatasetId)
          }}
        >
          <Segment
            key="shakespeare"
            label="Shakespeare"
            artwork={() => <FaFeatherAlt />}
          />
          <Segment key="recipes" label="Recipes" artwork={() => <FaUtensils />} />
        </SegmentedControl>
      </FadeIn>
    </Block>
  )

  const standardSummary =
    showTechnicalDetails && !isLoading && (
      <FadeIn>
        <FlexGrid flexGridColumnCount={[1, 1, 2]} flexGridColumnGap="scale600">
          <FlexGridItem>
            <FormControl
              label="Dataset preview"
              caption={() => (
                <>
                  Full text:{' '}
                  <StyledLink href={dataset?.textSourceURL}>
                    {datasetId === 'recipes' ? 'Recipe Box' : 'Tiny Shakespeare'}
                  </StyledLink>
                </>
              )}
              disabled={!dataset?.text}
            >
              <Textarea
                value={dataset?.text?.substring(0, 10_000) + '...'}
                rows={7}
                size={inputSize}
                readOnly
              />
            </FormControl>
          </FlexGridItem>

          <FlexGridItem>
            <Block display="flex" flexDirection="row" gridGap="scale600">
              <Block flex="1">
                <FormControl label="Dataset size">
                  <Count
                    count={dataset?.dataSize}
                    label="characters in total"
                    hierarchy="secondary"
                  />
                </FormControl>
              </Block>

              <Block flex="1">
                <FormControl label="Vocabulary size">
                  <Count
                    count={dataset?.vocabSize}
                    label="unique characters"
                    hierarchy="secondary"
                  />
                </FormControl>
              </Block>
            </Block>

            <FormControl
              label="Vocabulary"
              caption="Unique characters (including line breaks and spaces)"
              disabled={!dataset?.vocabulary}
            >
              <Textarea
                value={dataset?.vocabulary?.join('')}
                rows={3}
                size={inputSize}
                readOnly
              />
            </FormControl>
          </FlexGridItem>
        </FlexGrid>
      </FadeIn>
    )

  return (
    <Block>
      {segments}
      {loader}
      {error}
      {standardSummary}
    </Block>
  )
}
