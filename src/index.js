import React, { useCallback, useRef, useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import ObjectTracker from './ObjectTracker'

const ImageWithCanvas = ({ src, tracker }) => {
  const canvasRef = useRef()

  const handleLoad = useCallback(
    e => {
      canvasRef.current.width = e.target.width
      canvasRef.current.height = e.target.height
      const ctx = canvasRef.current.getContext('2d')
      ctx.lineWidth = 2
      ctx.strokeStyle = 'yellow'

      ctx.drawImage(e.target, 0, 0)

      const box = tracker.next(e.target)

      ctx.rect(box[0], box[1], box[2], box[3])
      ctx.stroke()
    },
    [tracker]
  )

  useEffect(() => {
    if (tracker) {
      const image = new Image()
      image.onload = handleLoad
      image.src = src
    }
  }, [handleLoad, src, tracker])

  return (
    <div>
      <canvas ref={canvasRef} />
    </div>
  )
}

const App = () => {
  const [tracker, setTracker] = useState()

  useEffect(() => {
    if (!tracker) {
      const image = new Image()
      image.onload = () => {
        const xmin = 256
        const ymin = 137
        const width = 84
        const height = 145
        const tracker = new ObjectTracker(image, [xmin, ymin, width, height])
        setTracker(tracker)
      }
      image.src = '/video/0001.jpg'
    }
  }, [tracker])

  const videos = [...new Array(1)].map(
    (_, i) => `/video/${(i + 1).toString().padStart(4, '0')}.jpg`
  )
  return (
    <div>
      {videos.map(imageSrc => (
        <ImageWithCanvas key={imageSrc} tracker={tracker} src={imageSrc} />
      ))}
    </div>
  )
}

ReactDOM.render(<App />, document.getElementById('root'))
