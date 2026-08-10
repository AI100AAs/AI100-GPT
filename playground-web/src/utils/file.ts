export const saveAsFile = (obj: Object, name: string) => {
  const jsonString = JSON.stringify(obj)
  const blob = new Blob([jsonString], { type: 'application/json' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)
  link.download = `${name}.json`
  link.href = url
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  setTimeout(() => URL.revokeObjectURL(url), 0)
}
