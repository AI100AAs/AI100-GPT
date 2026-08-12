import React from 'react'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'
import { Block } from 'baseui/block'
import { BackendId } from '../types/playground'
import { ProgressBar, SIZE } from 'baseui/progress-bar'
import { SegmentedControl, Segment } from 'baseui/segmented-control'
import { IoRocketSharp } from 'react-icons/io5'
import { Notification } from './shared/notification'
import { FadeIn } from './shared/fade'

type BackendProps = {
  backend: BackendId | undefined
  onChange?: (backend: BackendId) => Promise<void>
}

export function Backend(props: BackendProps) {
  const { backend, onChange = () => {} } = props

  const [errorMessage, setErrorMessage] = React.useState<string>()
  const [isLoading, setIsLoading] = React.useState<boolean>(false)

  const [hasWebGL, setHasWebGL] = React.useState<boolean>()
  const [hasWebGPU, setHasWebGPU] = React.useState<boolean>()

  const onBackendInit = async () => {
    setIsLoading(true)
    try {
      // How many GPU commands tfjs queues up before submitting them. This model
      // is small enough that no single operation keeps the GPU busy for long, so
      // the cost of generating a character is dominated by how many times the
      // command queue is handed to the driver -- roughly 500 operations go into
      // one character, which at the default of 15 means about thirty submissions
      // for a few microseconds of arithmetic each. Batching them more coarsely
      // spends less time talking to the driver. It only delays results, never
      // changes them.
      tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 64)

      // Walk the preference order until one backend actually activates.
      //
      // WASM is deliberately absent: it cannot train this model at all. The
      // backward pass through the embedding `gather` needs `UnsortedSegmentSum`,
      // which tfjs' WASM backend does not register, so training fails at the
      // first optimizer step with the forward pass having looked fine. WebGL is
      // the real fallback when WebGPU is unavailable.
      for (const candidate of TRAINING_BACKENDS) {
        if (candidate === 'webgpu' && !isWebGPUSupported()) continue
        if (candidate === 'webgl' && !isWebGLSupported()) continue
        try {
          // NOTE: `setBackend` resolves to `false` on failure rather than
          // throwing, so the return value has to be checked explicitly.
          const ok = await tf.setBackend(candidate)
          if (ok) break
          console.warn(`[tfjs] ${candidate} reported as supported but could not be activated`)
        } catch (err) {
          // Adapter/context requests can still fail on paper-supported devices.
          console.warn(`[tfjs] ${candidate} activation threw, trying next`, err)
        }
      }
      await tf.ready()

      // `tf.getBackend()` is the backend that is actually active. (Reading the
      // registry instead reports whichever backend happened to register first.)
      const activeBackend = tf.getBackend()
      console.info('[tfjs] active backend:', activeBackend)
      await onChange(activeBackend as BackendId)
    } catch (err) {
      setErrorMessage((err as Error).message)
    } finally {
      setIsLoading(false)
    }
  }

  const onBackendChange = async (nextBackend: BackendId) => {
    setIsLoading(true)
    setErrorMessage(undefined)
    try {
      const success = await tf.setBackend(nextBackend)
      await tf.ready()
      if (!success) {
        throw new Error(`Cannot set a ${nextBackend} backend`)
      }
      // Report what is actually active, not what was requested.
      const activeBackend = tf.getBackend()
      console.info('[tfjs] active backend:', activeBackend)
      onChange(activeBackend as BackendId)
    } catch (err) {
      setErrorMessage((err as Error).message)
    }
    setIsLoading(false)
  }

  React.useEffect(() => {
    setHasWebGL(isWebGLSupported())
    setHasWebGPU(isWebGPUSupported())
    onBackendInit()
  }, [])

  const error = errorMessage && (
    <Notification kind="negative">{errorMessage}</Notification>
  )

  const loader = isLoading && (
    <FadeIn>
      <ProgressBar
        infinite
        size={SIZE.small}
        getProgressLabel={() => 'Setting the TensorFlow backend...'}
        showLabel
        overrides={{ BarContainer: { style: { marginLeft: 0, marginRight: 0 } } }}
      />
    </FadeIn>
  )

  const segments = (
    <FadeIn>
      <SegmentedControl
        activeKey={backend}
        disabled={isLoading}
        onChange={({ activeKey }) => {
          onBackendChange(activeKey as BackendId)
        }}
      >
        {(Object.keys(BACKENDS) as (keyof typeof BACKENDS)[]).map((backendId) => {
          const disabled =
            (backendId === 'webgpu' && !hasWebGPU) ||
            (backendId === 'webgl' && !hasWebGL)
          return (
            <Segment
              key={backendId}
              disabled={disabled}
              label={BACKENDS[backendId].label}
              description={BACKENDS[backendId].description}
            />
          )
        })}
      </SegmentedControl>
    </FadeIn>
  )

  const noGPUSupportWarning =
    !hasWebGL && !hasWebGPU ? (
      <Notification kind="warning">
        Looks like your browser supports neither WebGPU nor WebGL. Training will fall
        back to the CPU and will be slow.
      </Notification>
    ) : null

  const slowCPUWarning = ['cpu', 'webgl'].includes(backend || '') ? (
    <Notification kind="warning">
      Training on <b>{backend?.toUpperCase()}</b> might be slow. The recommended setup is
      to use a device and browser that support <b>WebGPU</b>.
    </Notification>
  ) : null

  return (
    <Block>
      {segments}
      {error}
      {noGPUSupportWarning || slowCPUWarning}
      {loader}
    </Block>
  )
}

// Backends that can actually run a training step, best first. WASM is excluded:
// it has no `UnsortedSegmentSum` kernel, so the backward pass through the
// embedding lookup throws and training stops at the first optimizer step.
export const TRAINING_BACKENDS: BackendId[] = ['webgpu', 'webgl', 'cpu']


export const BACKENDS = {
  webgl: {
    label: 'WebGL',
    description: (
      <SegmentDescription>
        <IoRocketSharp size={13} />
      </SegmentDescription>
    ),
  },
  webgpu: {
    label: 'WebGPU',
    description: (
      <SegmentDescription>
        <IoRocketSharp size={13} />
        <IoRocketSharp size={13} />
      </SegmentDescription>
    ),
  },
}

function SegmentDescription({ children }: { children: React.ReactNode }) {
  return (
    <Block
      display="flex"
      flexDirection="row"
      justifyContent="center"
      marginTop="8px"
      gridGap="2px"
    >
      {children}
    </Block>
  )
}

function isWebGLSupported() {
  try {
    const canvas = document.createElement('canvas')
    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')))
  } catch (err) {
    return false
  }
}

function isWebGPUSupported() {
  try {
    return !!navigator.gpu
  } catch (err) {
    return false
  }
}
