// Every URL the app builds -- weights, adapters, dataset text, the header's
// home link -- is relative to wherever its own index.html was loaded from.
// That is what lets the same build run unmodified at a domain root, inside a
// GitHub Pages project subpath, or embedded in an iframe on someone else's
// site: the browser resolves "./weights/" against the current document, not
// against a path baked in at build time.
export const BASE_PATH = '.'
export const MODEL_WEIGHTS_BASE_URL = `${BASE_PATH}/weights/`
