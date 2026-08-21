import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { Playground } from './components/playground'
import { Layout } from './components/shared/layout'

// A wildcard match rather than a fixed path: the app has one screen and no
// internal routes, and it needs to render the same way no matter what prefix
// the surrounding site (a GitHub Pages subpath, an iframe host page, ...)
// puts in front of it.
const router = createBrowserRouter([
  {
    path: '*',
    element: <Playground />,
  },
])

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement)
root.render(
  <Layout>
    <RouterProvider router={router} />
  </Layout>,
)
