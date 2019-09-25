import * as tf from '@tensorflow/tfjs'

const EPSILON = 1e-7
const SIGMA = 100
const PRETRAINING_STEPS = 0
const LEARNING_RATE = 0.125

const rgbToGrayscale = image => {
  // Reference for converting between RGB and grayscale.
  // https://en.wikipedia.org/wiki/Luma_%28video%29
  const rgbWeights = tf.tensor1d([0.2989, 0.587, 0.114])
  return tf.sum(image.mul(rgbWeights), 2)
}

const gauss = (width, height, sigma, center) => {
  const gaussBuffer = tf.zeros([height, width]).bufferSync()

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = 1 / (2 * Math.pow(sigma, 2))
      const fin = Math.exp(
        -(a * Math.pow(x - center[1], 2) + a * Math.pow(y - center[0], 2))
      )
      gaussBuffer.set(fin, y, x)
    }
  }

  return gaussBuffer.toTensor()
}

const fft2 = tensor2d => {
  let realPart
  let imagPart
  if (tensor2d.dtype.includes('complex')) {
    realPart = tf.real(tensor2d).asType('float32') // weird bug?
    imagPart = tf.imag(tensor2d).asType('float32')
  } else {
    realPart = tensor2d
    imagPart = tf.zerosLike(tensor2d)
  }

  const initReal = tf.unstack(realPart, 1)
  const initImag = tf.unstack(imagPart, 1)

  const unstackedColZip = initReal.map((real, i) => {
    const imag = initImag[i]
    const x = tf.complex(real, imag)
    const res = x.fft()
    return [tf.real(res).as1D(), tf.imag(res).as1D()]
  })

  const unstackedRealCol = unstackedColZip.map(x => x[0])
  const unstackedImagCol = unstackedColZip.map(x => x[1])

  const stackedReal = tf.stack(unstackedRealCol, 1)
  const stackedImag = tf.stack(unstackedImagCol, 1)

  const unstackedRealRow = tf.unstack(stackedReal)
  const unstackedImagRow = tf.unstack(stackedImag)

  const unstackedRowZip = unstackedRealRow.map((real, i) => {
    const imag = unstackedImagRow[i]
    const x = tf.complex(real, imag)
    const res = x.fft()
    return [tf.real(res).as1D(), tf.imag(res).as1D()]
  })

  const unstackedRealRowUnzipped = unstackedRowZip.map(x => x[0])
  const unstackedImagRowUnzipped = unstackedRowZip.map(x => x[1])

  const stackedRealFinal = tf.stack(unstackedRealRowUnzipped)
  const stackedImagFinal = tf.stack(unstackedImagRowUnzipped)

  return tf.complex(stackedRealFinal, stackedImagFinal)
}

const conjugate = complex => {
  return tf.complex(tf.real(complex), tf.imag(complex).mul(-1))
}

const hanning = M => {
  const numberLine = tf.linspace(0, M - 1, M)
  const intermediate = tf.cos(numberLine.mul(2 * Math.PI).div(M - 1))
  return tf.scalar(0.5).sub(intermediate.mul(0.5))
}

const hanningWindow = (width, height) => {
  const col = hanning(width)
  const row = hanning(height).reshape([-1, 1])
  return col.mul(row)
}

const preprocessImage = image => {
  const [height, width] = image.shape
  const logOfImage = tf.log1p(image) // log(image + 1)
  const stdOfImage = tf.moments(logOfImage).variance.sqrt()
  const normalizedImage = logOfImage
    .sub(tf.mean(logOfImage))
    .div(stdOfImage.add(EPSILON))

  const window = hanningWindow(width, height)

  return normalizedImage.mul(window)
}

const randomWarp = image => {
  // TODO: Random scale and rotations.
  // const a = -180/16
  // const b = 180/16
  // const r = a + (b-a) * Math.random()

  // const scale = 1 - 0.1 + 0.2 * Math.random()
  // const img = imresize(imresize(imrotate(image, r), scale), [sz(1) sz(2)]);
  return image
}

const normalize = tensor =>
  tensor.sub(tf.min(tensor)).div(tf.max(tensor).sub(tf.min(tensor)))

const findIndex2d = (matrix, val) => {
  return matrix.reduce((acc, row, y) => {
    row.forEach((item, x) => {
      if (item === val) {
        acc.push([y, x])
      }
    })
    return acc
  }, [])
}

// tfjs multiplying complex numbers doesn't work.
const complexMul = (a, b) => tf.complex(tf.real(a).mul(b), tf.imag(a).mul(b))
const complexDiv = (a, b) => {
  const aReal = tf.real(a)
  const aImag = tf.imag(a)
  const bReal = tf.real(b)
  const bImag = tf.imag(b)
  const denom = bImag.mul(bImag).add(bReal.mul(bReal))
  const real = aReal
    .mul(bReal)
    .add(aImag.mul(bImag))
    .div(denom)
  const imag = bReal
    .mul(aImag)
    .sub(aReal.mul(bImag))
    .div(denom)
  return tf.complex(real, imag)
}

class ObjectTracker {
  constructor(frame, [xmin, ymin, width, height]) {
    this.initialRect = [xmin, ymin, width, height]

    const image = tf.browser.fromPixels(frame)
    const greyscaleImage = rgbToGrayscale(image)

    const center = [ymin + height / 2, xmin + width / 2]
    const gaussTensor = gauss(frame.width, frame.height, SIGMA, center)

    const gaussCrop = gaussTensor.slice([ymin, xmin], [height, width])

    const imageCrop = greyscaleImage.slice([ymin, xmin], [height, width])
    const processedImage = preprocessImage(imageCrop)

    // tf.complex(gaussCrop.flatten(), gaussCrop.flatten().zerosLike())
    //   .fft()
    //   .print()

    const gaussFourier = fft2(gaussCrop)
    gaussFourier.print()

    const processedImageFourier = fft2(processedImage)

    this.Ai = gaussFourier.mul(conjugate(processedImageFourier))
    this.Bi = processedImageFourier.mul(conjugate(processedImageFourier))

    for (let i = 0; i < PRETRAINING_STEPS; i++) {
      const processedImage = preprocessImage(randomWarp(imageCrop))
      const processedImageFourier = fft2(processedImage)
      this.Ai = this.Ai.add(gaussFourier.mul(conjugate(processedImageFourier)))
      this.Bi = this.Bi.add(
        processedImageFourier.mul(conjugate(processedImageFourier))
      )
    }

    this.Ai = complexMul(this.Ai, LEARNING_RATE)
    this.Bi = complexMul(this.Bi, LEARNING_RATE)
  }

  next = frame => {
    const image = tf.browser.fromPixels(frame)
    const greyscaleImage = rgbToGrayscale(image)

    const [xmin, ymin, width, height] = this.initialRect
    const imageCrop = greyscaleImage.slice([ymin, xmin], [height, width])
    const processedImage = preprocessImage(imageCrop)

    const Hi = complexDiv(this.Ai, this.Bi)

    const gi = tf.real(fft2(complexMul(Hi, fft2(processedImage))))
    const normalizedGi = normalize(gi)

    const maxValue = tf.max(normalizedGi).dataSync()[0]
    const positions = findIndex2d(normalizedGi.arraySync(), maxValue)
    const positionsTransposed = tf.tensor2d(positions).transpose()

    const dy = tf
      .mean(positionsTransposed.slice(0))
      .sub(normalizedGi.shape[0] / 2)
      .dataSync()[0]
    const dx = tf
      .mean(positionsTransposed.slice(1))
      .sub(normalizedGi.shape[1] / 2)
      .dataSync()[0]

    return [xmin + dx, ymin + dy, width, height]
  }
}

export default ObjectTracker
