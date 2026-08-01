import React from 'react'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import '@tensorflow/tfjs-backend-webgpu'
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm'
import { Block } from 'baseui/block'
import { BackendId } from '../types/playground'
import { ProgressBar, SIZE } from 'baseui/progress-bar'
import { SegmentedControl, Segment } from 'baseui/segmented-control'
import { BASE_PATH } from '../config/links'
import { IoRocketSharp } from 'react-icons/io5'
import { ReactComponent as TurtleIcon } from '../assets/turtle.svg'
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
  const [hasWASM, setHasWASM] = React.useState<boolean>()

  const onBackendInit = async () => {
    setIsLoading(true)
    try {
      // For WASM backend, set the path to the public folder where `.wasm` files are located
      setWasmPaths(BASE_PATH + '/')

      // Prefer WebGPU explicitly: it is by far the fastest backend for training,
      // and relying on tfjs' default pick can silently land us on WASM.
      if (isWebGPUSupported()) {
        try {
          // NOTE: `setBackend` resolves to `false` on failure rather than
          // throwing, so the return value has to be checked explicitly.
          const ok = await tf.setBackend('webgpu')
          if (!ok) {
            console.warn('[tfjs] WebGPU was reported as supported but could not be activated')
          }
        } catch (err) {
          // Adapter request can still fail on paper-supported devices; fall
          // through and let tfjs pick whatever else is available.
          console.warn('[tfjs] WebGPU activation threw, falling back', err)
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
    setHasWASM(isWASMSupported())
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
            (backendId === 'wasm' && !hasWASM)
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
        Looks like your browser doesn't support neither WebGPU nor WebGL. Training on CPU
        or WASM might be slow.
      </Notification>
    ) : null

  const slowCPUWarning = ['cpu', 'wasm', 'webgl'].includes(backend || '') ? (
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

// TODO: Re-enable cpu and webgl backends once performance is acceptable
export const BACKENDS = {
  // cpu: { label: 'CPU', description: <SegmentDescription><TurtleIcon width="16" /><TurtleIcon width="16" /></SegmentDescription> },
  wasm: {
    label: 'WASM',
    description: (
      <SegmentDescription>
        <TurtleIcon width="16" />
      </SegmentDescription>
    ),
  },
  // webgl: { label: 'WebGL', description: <SegmentDescription><IoRocketSharp size={13} /></SegmentDescription> },
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

function isWASMSupported() {
  return typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function'
}

