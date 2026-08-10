import React from 'react'
import { Dataset as DatasetT, Model as ModelT, ModelVariant } from '@gpt/model'
import { Backend } from './backend'
import { FinetuneCorpus } from './finetune'
import { CompareGenerators } from './compare'
import { StyleFile } from './style-file'
import { Tabs, Tab, FILL } from 'baseui/tabs-motion'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { BASE_DATASETS, BackendId, BaseDatasetId } from '../types/playground'
import { Step } from './shared/step'
import { Dataset } from './dataset'
import { Model } from './model'
import { Footer } from './shared/footer'
import { Trainer } from './trainer'
import { WINDOW_PADDING_HORIZONTAL } from '../config/theme'
import { Block } from 'baseui/block'
import { Generator } from './generator'
import { Debugger } from './debugger'
import { saveAsFile } from '../utils/file'
import { Button, KIND } from 'baseui/button'
import { RiDownloadLine } from 'react-icons/ri'
import { FormControl } from 'baseui/form-control'
import { Notification } from './shared/notification'

export function Playground() {
  // The technical view (backend picker, dataset measurements, training controls,
  // debugger) is kept in the code but hidden from the student interface. Flip
  // this to true to bring it back.
  const showTechnicalDetails = false
  const [backend, setBackend] = React.useState<BackendId>()

  // Models are always built with adapters attached. Before training they are an
  // exact no-op, so the pretrained model generates unchanged; afterwards the same
  // model carries the learner's style. That avoids rebuilding between steps.
  const loraRank = 8
  // The learner's own text, encoded with the base model's character set.
  const [corpus, setCorpus] = React.useState<DatasetT>()
  const [activeTab, setActiveTab] = React.useState<string>('explore')
  const [hasTrainedStyle, setHasTrainedStyle] = React.useState(false)
  const [isTrainingStyle, setIsTrainingStyle] = React.useState(false)
  const [styleDraftRevision, setStyleDraftRevision] = React.useState(0)
  const [styleMessage, setStyleMessage] = React.useState<{
    kind: 'positive' | 'negative'
    text: string
  }>()
  // Undefined means "same as training" -- generation then runs in place with no
  // backend switch and no model rebuild.
  const [inferenceBackend, setInferenceBackend] = React.useState<BackendId>()

  const [dataset, setDataset] = React.useState<DatasetT>()
  const [datasetId, setDatasetId] = React.useState<BaseDatasetId>('shakespeare')
  const baseDatasetId = datasetId
  // The selected dataset has already loaded the exact vocabulary used by its
  // checkpoint. Reuse it for adaptation instead of fetching the same text a
  // second time; a failed duplicate request used to leave training permanently
  // disabled even though generation was ready.
  const baseVocabulary = dataset?.vocabulary

  const [model, setModel] = React.useState<ModelT>()
  const [modelVariant, setModelVariant] = React.useState<ModelVariant>()

  const onCorpusChange = React.useCallback((nextCorpus: DatasetT | undefined) => {
    setCorpus((previousCorpus) => {
      if (previousCorpus !== nextCorpus) previousCorpus?.dispose?.()
      return nextCorpus
    })
  }, [])

  const onBackendChange = async (backend: BackendId) => {
    model?.dispose?.()
    setModel(undefined)

    dataset?.dispose?.()
    setDataset(undefined)

    setBackend(backend)
  }

  const onDatasetChange = async (nextDataset: DatasetT, nextDatasetId: BaseDatasetId) => {
    model?.dispose?.()
    setModel(undefined)

    // Never leave training data or vocabulary from the previous base model
    // attached while the new checkpoint is loading.
    onCorpusChange(undefined)
    setHasTrainedStyle(false)
    setStyleMessage(undefined)

    dataset?.dispose?.()
    setDataset(nextDataset)
    setDatasetId(nextDatasetId)
  }

  const onModelChange = async (nextModel: ModelT, nextModelVariant: ModelVariant) => {
    model?.dispose?.()
    setModel(nextModel)
    setModelVariant(nextModelVariant)
  }

  const onDownloadStyle = () => {
    try {
      if (!model?.getLoRAWeights || !modelVariant || !hasTrainedStyle) {
        throw new Error('Train or load a style before saving it.')
      }
      const weights = model.getLoRAWeights()
      saveAsFile(
        {
          format: 'teachable-lm-style',
          version: 1,
          base: baseDatasetId,
          modelVariant,
          blockSize: model.params.blockSize,
          vocabSize: model.params.vocabSize,
          ...weights,
        },
        `my-style--${baseDatasetId}`,
      )
      setStyleMessage({ kind: 'positive', text: 'Style saved to your downloads.' })
    } catch (err) {
      setStyleMessage({ kind: 'negative', text: (err as Error).message })
    }
  }

  const onUploadStyle = (file: File) => {
    setStyleMessage(undefined)
    const reader = new FileReader()
    reader.onload = () => {
      try {
        if (!model?.setLoRAWeights || !modelVariant) {
          throw new Error('The model is still loading. Try again in a moment.')
        }
        const parsed = JSON.parse(String(reader.result))
        if (parsed?.format !== 'teachable-lm-style' || parsed?.version !== 1) {
          throw new Error('That file is not a saved style from this playground.')
        }
        if (parsed.base !== baseDatasetId) {
          const savedBase = BASE_DATASETS[parsed.base as BaseDatasetId]?.label ?? parsed.base
          throw new Error(`That style belongs to the ${savedBase} model. Switch to it first.`)
        }
        if (parsed.modelVariant && parsed.modelVariant !== modelVariant) {
          throw new Error('That style was saved from a different model size.')
        }
        if (!Array.isArray(parsed.adapters) || !parsed.adapters.length) {
          throw new Error('That style file has no trained weights.')
        }
        const validNumbers =
          Number.isFinite(parsed.rank) &&
          Number.isFinite(parsed.alpha) &&
          parsed.adapters.every(
            (adapter: { a?: unknown; b?: unknown }) =>
              Array.isArray(adapter?.a) &&
              Array.isArray(adapter?.b) &&
              adapter.a.every(Number.isFinite) &&
              adapter.b.every(Number.isFinite),
          )
        if (!validNumbers) {
          throw new Error('That style file contains invalid weights.')
        }
        model.setLoRAWeights(parsed)
        model.setLoRAEnabled?.(true)
        setHasTrainedStyle(true)
        setStyleMessage({ kind: 'positive', text: 'Style loaded. Open Compare to try it.' })
      } catch (err) {
        setStyleMessage({ kind: 'negative', text: (err as Error).message })
      }
    }
    reader.onerror = () => {
      setStyleMessage({ kind: 'negative', text: 'Could not read that style file.' })
    }
    reader.readAsText(file)
  }

  return (
    <>
      <Block
        paddingLeft={WINDOW_PADDING_HORIZONTAL}
        paddingRight={WINDOW_PADDING_HORIZONTAL}
      >
        <Block
          marginBottom="scale800"
          padding="scale600"
          backgroundColor="backgroundSecondary"
          $style={{ borderRadius: '8px', border: '1px solid rgba(127, 127, 127, 0.22)' }}
        >
          <Block
            marginBottom="scale200"
            color="contentPrimary"
            $style={{ fontSize: '20px', fontWeight: 700, lineHeight: '28px' }}
          >
            How this playground works
          </Block>
          <Block color="contentSecondary" $style={{ fontSize: '16px', lineHeight: '24px' }}>
            These are very small models—orders of magnitude smaller than current chatbot
            models—trained on two different text styles. Pick one and watch it continue a
            phrase one character at a time.
            Then give it a few lines of your own writing and see it pick up your style while
            keeping the voice it learned first. The model runs entirely in your browser, on
            your device: your writing and generated text never leave it.
          </Block>
        </Block>

        <Block $style={{ display: 'none' }}>
          <Backend backend={backend} onChange={onBackendChange} />
          <Model
            model={model}
            backend={backend}
            vocabSize={dataset?.vocabSize}
            onChange={onModelChange}
            onDownloadModelWeights={() => {}}
            showTechnicalDetails={showTechnicalDetails}
            datasetId={datasetId}
            tuning="lora"
            loraRank={loraRank}
            baseDatasetId={baseDatasetId}
            usesBaseVocabulary={false}
          />
        </Block>

        <Tabs
          activeKey={activeTab}
          onChange={({ activeKey }) => {
            if (!isTrainingStyle) setActiveTab(String(activeKey))
          }}
          activateOnFocus
          fill={FILL.fixed}
          renderAll
          overrides={{
            Root: { style: { marginBottom: '24px' } },
            TabList: {
              style: ({ $theme }) => ({
                backgroundColor: $theme.colors.backgroundSecondary,
                borderRadius: '12px',
                padding: '4px',
                gap: '4px',
              }),
            },
            TabHighlight: { style: { display: 'none' } },
            TabBorder: { style: { display: 'none' } },
          }}
        >
          <Tab
            key="explore"
            title="Explore"
            disabled={isTrainingStyle}
            overrides={TAB_OVERRIDES}
          >
            <Step title="Choose a style">
              <Dataset
                dataset={dataset}
                onChange={onDatasetChange}
                backend={backend}
                showTechnicalDetails={showTechnicalDetails}
              />
            </Step>

            <Step title="Generate text">
              <Generator
                key={`explore-${baseDatasetId}`}
                model={model}
                dataset={dataset}
                modelVariant={modelVariant}
                datasetId={datasetId}
                inferenceBackend={inferenceBackend}
                useAdapters={false}
                showTechnicalDetails={showTechnicalDetails}
              />
            </Step>
          </Tab>

          <Tab key="teach" title="Teach it your style" overrides={TAB_OVERRIDES}>
            <Step title="Your own words">
              <FinetuneCorpus
                baseDatasetId={baseDatasetId}
                baseVocabulary={baseVocabulary}
                corpus={corpus}
                onChange={onCorpusChange}
                onTextEdited={() => {
                  setHasTrainedStyle(false)
                  setStyleMessage(undefined)
                  setStyleDraftRevision((revision) => revision + 1)
                }}
                disabled={isTrainingStyle}
                blockSize={model?.params.blockSize}
              />
              <Block marginTop="scale700">
                <Trainer
                  key={`trainer-${baseDatasetId}-${styleDraftRevision}`}
                  model={model}
                  dataset={corpus}
                  simplified
                  onTrainingStateChange={(training) => {
                    setIsTrainingStyle(training)
                    if (training) {
                      setHasTrainedStyle(false)
                      setStyleMessage(undefined)
                    }
                  }}
                  onTrainingComplete={() => setHasTrainedStyle(true)}
                />
              </Block>

              <FlexGrid
                flexGridColumnCount={[1, 1, 2]}
                flexGridColumnGap="scale600"
                marginTop="scale700"
              >
                <FlexGridItem>
                  <FormControl
                    label="Save a style"
                    caption="Download the style you trained so you can use it again later."
                  >
                    <Button
                      kind={KIND.secondary}
                      disabled={!hasTrainedStyle || !model}
                      onClick={onDownloadStyle}
                      startEnhancer={() => <RiDownloadLine />}
                      overrides={{ Root: { style: { width: '100%' } } }}
                    >
                      Save trained style
                    </Button>
                  </FormControl>
                </FlexGridItem>
                <FlexGridItem>
                  <StyleFile
                    onUpload={onUploadStyle}
                    baseLabel={BASE_DATASETS[baseDatasetId].label}
                    disabled={isTrainingStyle}
                  />
                </FlexGridItem>
              </FlexGrid>

              {styleMessage && (
                <Block marginTop="scale400">
                  <Notification kind={styleMessage.kind}>{styleMessage.text}</Notification>
                </Block>
              )}

            </Step>
          </Tab>

          <Tab
            key="compare"
            title="Compare"
            disabled={isTrainingStyle}
            overrides={TAB_OVERRIDES}
          >
            <Step title="Before and after">
              <CompareGenerators
                key={`compare-${baseDatasetId}`}
                model={model}
                dataset={dataset}
                baseDatasetId={baseDatasetId}
                hasTrainedStyle={hasTrainedStyle}
                onOpenTraining={() => setActiveTab('teach')}
              />
            </Step>
          </Tab>

        </Tabs>

      </Block>

      <Footer />
    </>
  )
}

// Pill-shaped tabs: the selected one gets a raised surface rather than the
// default underline, which reads better above card-based content.
const TAB_OVERRIDES = {
  Tab: {
    style: ({ $theme, $isActive }: { $theme: any; $isActive: boolean }) => ({
      flexGrow: 1,
      textAlign: 'center' as const,
      borderRadius: '9px',
      paddingTop: '9px',
      paddingBottom: '9px',
      fontSize: '14px',
      fontWeight: $isActive ? 600 : 500,
      color: $isActive ? $theme.colors.contentPrimary : $theme.colors.contentSecondary,
      backgroundColor: $isActive ? $theme.colors.backgroundPrimary : 'transparent',
      boxShadow: $isActive ? '0 1px 2px rgba(16, 24, 40, 0.06)' : 'none',
      ':hover': { backgroundColor: $isActive ? $theme.colors.backgroundPrimary : $theme.colors.backgroundTertiary },
    }),
  },
  TabPanel: { style: { paddingLeft: 0, paddingRight: 0, paddingBottom: 0 } },
}
