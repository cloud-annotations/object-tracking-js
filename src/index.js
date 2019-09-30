import React, { useCallback, useRef, useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import objectTracker from './object-tracking'

const ImageWithCanvas = ({ src, tracker }) => {
  const canvasRef = useRef()

  const handleLoad = useCallback(
    async e => {
      canvasRef.current.width = e.target.width
      canvasRef.current.height = e.target.height
      const ctx = canvasRef.current.getContext('2d')
      ctx.lineWidth = 2
      ctx.strokeStyle = 'yellow'

      ctx.drawImage(e.target, 0, 0)

      const box = await tracker.next(e.target)

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
  const [currentIndex, setCurrentIndex] = useState(0)

  const debugCanvas = useRef()

  const videos = [...new Array(376)].map(
    (_, i) => `/video/${(i + 1).toString().padStart(4, '0')}.jpg`
  )

  useEffect(() => {
    if (!tracker) {
      const image = new Image()
      image.onload = () => {
        const xmin = 256
        const ymin = 137
        const width = 84
        const height = 145
        const tracker = objectTracker.init(image, [xmin, ymin, width, height])
        setTracker(tracker)
      }
      image.src = videos[0]
    }
  }, [tracker, videos])

  useEffect(() => {
    const handleClick = e => {
      if (e.code === 'ArrowRight') {
        setCurrentIndex(i => i + 1)
      }
    }
    document.addEventListener('keydown', handleClick)
    return () => {
      document.removeEventListener('keydown', handleClick)
    }
  }, [])

  return (
    <div>
      <canvas ref={debugCanvas}></canvas>
      <ImageWithCanvas tracker={tracker} src={videos[currentIndex]} />
    </div>
  )
}

ReactDOM.render(<App />, document.getElementById('root'))
