import React from 'react'
import { Dataset as DatasetT, Model as ModelT, ModelVariant } from '@gpt/model'
import { Backend } from './backend'
import { BackendId, DatasetId } from '../types/playground'
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
import { Checkbox, STYLE_TYPE } from 'baseui/checkbox'

export function Playground() {
  const [showTechnicalDetails, setShowTechnicalDetails] = React.useState(false)
  const [backend, setBackend] = React.useState<BackendId>()

  const [dataset, setDataset] = React.useState<DatasetT>()
  const [datasetId, setDatasetId] = React.useState<DatasetId>()

  const [model, setModel] = React.useState<ModelT>()
  const [modelVariant, setModelVariant] = React.useState<ModelVariant>()

  const onBackendChange = async (backend: BackendId) => {
    model?.dispose?.()
    setModel(undefined)

    dataset?.dispose?.()
    setDataset(undefined)

    setBackend(backend)
  }

  const onDatasetChange = async (nextDataset: DatasetT, nextDatasetId: DatasetId) => {
    model?.dispose?.()
    setModel(undefined)

    dataset?.dispose?.()
    setDataset(nextDataset)
    setDatasetId(nextDatasetId)
  }

  const onModelChange = async (nextModel: ModelT, nextModelVariant: ModelVariant) => {
    model?.dispose?.()
    setModel(nextModel)
    setModelVariant(nextModelVariant)
  }

  const onDownloadModelWeights = async () => {
    const weights = await model?.getWeights?.()
    if (weights && modelVariant) {
      const fileName = modelVariant + '--' + datasetId
      saveAsFile(weights, fileName)
    }
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
            Explore a small language model trained on Shakespeare. Choose a model, optionally
            train it further, then give it a starting phrase and watch it continue one character
            at a time. Use this space to see how a model&apos;s size, training, and input context
            shape what it predicts next. Everything runs in your browser.
          </Block>
          <Block
            marginTop="scale500"
            paddingTop="scale400"
            $style={{ borderTop: '1px solid rgba(127, 127, 127, 0.22)' }}
          >
            <Checkbox
              checked={showTechnicalDetails}
              checkmarkType={STYLE_TYPE.toggle}
              onChange={(event) => setShowTechnicalDetails(event.currentTarget.checked)}
            >
              Technical details
            </Checkbox>
            <Block
              marginTop="scale100"
              color="contentSecondary"
              $style={{ fontSize: '14px', lineHeight: '20px' }}
            >
              Show backend options, dataset measurements, architecture, training controls,
              and debugging tools.
            </Block>
          </Block>
        </Block>

        {showTechnicalDetails ? (
          <Step title="1. TensorFlow Backend">
            <Backend backend={backend} onChange={onBackendChange} />
          </Step>
        ) : (
          <Block $style={{ display: 'none' }}>
            <Backend backend={backend} onChange={onBackendChange} />
          </Block>
        )}

        <Step title={showTechnicalDetails ? '2. Dataset' : '1. Choose a dataset'}>
          <Dataset
            dataset={dataset}
            onChange={onDatasetChange}
            backend={backend}
            showTechnicalDetails={showTechnicalDetails}
          />
        </Step>

        <Step title={showTechnicalDetails ? '3. GPT Model Size' : '2. Choose a model'}>
          <Model
            model={model}
            backend={backend}
            vocabSize={dataset?.vocabSize}
            onChange={onModelChange}
            onDownloadModelWeights={onDownloadModelWeights}
            showTechnicalDetails={showTechnicalDetails}
            datasetId={datasetId}
          />
        </Step>

        {(showTechnicalDetails || datasetId === 'custom') && (
          <Step title={showTechnicalDetails ? '4. Model Training' : '3. Train your model'}>
            <Trainer
              model={model}
              dataset={dataset}
              simplified={!showTechnicalDetails}
            />
          </Step>
        )}

        <Step
          title={
            showTechnicalDetails
              ? '5. Generation (Prediction)'
              : datasetId === 'custom'
                ? '4. Generate text'
                : '3. Generate text'
          }
        >
          <Generator
            model={model}
            dataset={dataset}
            modelVariant={modelVariant}
            datasetId={datasetId}
            showTechnicalDetails={showTechnicalDetails}
          />
        </Step>

        {showTechnicalDetails && (
          <Step title="Debug" accordion closed>
            <Debugger model={model} dataset={dataset} backend={backend} />
          </Step>
        )}
      </Block>

      <Footer />
    </>
  )
}
