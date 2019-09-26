import * as tf from '@tensorflow/tfjs'

const EPSILON = 1e-7
const SIGMA = 100
const PRETRAINING_STEPS = 0
const LEARNING_RATE = 0.125

const rgbToGrayscale = image => {
  // Reference for converting between RGB and grayscale.
  // https://en.wikipedia.org/wiki/Luma_%28video%29
  const rgbWeights = tf.tensor1d([0.2989, 0.587, 0.114])
  return tf.sum(image.mul(rgbWeights), 2) // broadcast across the image.
}

const gauss = (width, height, sigma, center) => {
  const gaussBuffer = tf.zeros([height, width]).bufferSync()

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = 1 / (2 * sigma)
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
    realPart = tf.real(tensor2d)
    imagPart = tf.imag(tensor2d)
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
  return tf.complex(tf.real(complex), tf.imag(complex).mul(-1)) // broadcast scalar
}

const hanning = M => {
  const numberLine = tf.linspace(0, M - 1, M)
  const intermediate = tf.cos(numberLine.mul(2 * Math.PI).div(M - 1)) // multiplying by a scalar
  return tf.scalar(0.5).sub(intermediate.mul(0.5)) // multiplying by a scalar
}

const hanningWindow = (width, height) => {
  const col = hanning(width)
  const row = hanning(height).reshape([-1, 1])
  return col.mul(row) // broadcast
}

const preprocessImage = image => {
  const [height, width] = image.shape
  const logOfImage = tf.log1p(image) // log(image + 1)
  const stdOfImage = tf.moments(logOfImage).variance.sqrt()
  const normalizedImage = logOfImage
    .sub(tf.mean(logOfImage))
    .div(stdOfImage.add(EPSILON))

  const window = hanningWindow(width, height)

  return normalizedImage.mulStrict(window)
}

const randomWarp = image => {
  // TODO: Random scale and rotations.
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

// tfjs multiplying complex numbers doesn't work well.
const complexMul = (a, b) => {
  // CASES:
  if (a.dtype && b.dtype) {
    // complex * complex
    if (a.dtype.includes('complex') && b.dtype.includes('complex')) {
      const aReal = tf.real(a)
      const aImag = tf.imag(a)
      const bReal = tf.real(b)
      const bImag = tf.imag(b)
      const r1r2 = aReal.mulStrict(bReal)
      const r1i2 = aReal.mulStrict(bImag)
      const i1r2 = aImag.mulStrict(bReal)
      const i1i2 = aImag.mulStrict(bImag)

      const real = r1r2.sub(i1i2)
      const imag = r1i2.add(i1r2)
      return tf.complex(real, imag)
    }

    // complex * tensor
    if (
      a.dtype.includes('complex') &&
      b.dtype.includes('float') &&
      b.rankType >= 1
    ) {
      return tf.complex(tf.real(a).mulStrict(b), tf.imag(a).mulStrict(b))
    }

    // tensor * complex
    if (
      a.dtype.includes('float') &&
      a.rankType >= 1 &&
      b.dtype.includes('complex')
    ) {
      return tf.complex(tf.real(b).mulStrict(a), tf.imag(b).mulStrict(a))
    }
  }

  // complex * scalar/num
  if (a.dtype && a.dtype.includes('complex')) {
    return tf.complex(tf.real(a).mul(b), tf.imag(a).mul(b)) // allow broadcast
  }

  // tensor * tensor
  // tensor * scalar
  return a.mul(b) // allow broadcast

  // ignore these cases for now...
  // scalar * complex
  // scalar * tensor
  // scalar * scalar
}

const complexDiv = (a, b) => {
  const aReal = tf.real(a)
  const aImag = tf.imag(a)
  const bReal = tf.real(b)
  const bImag = tf.imag(b)
  const denom = bImag.mulStrict(bImag).add(bReal.mulStrict(bReal))
  const real = aReal
    .mulStrict(bReal)
    .add(aImag.mulStrict(bImag))
    .div(denom)
  const imag = bReal
    .mulStrict(aImag)
    .sub(aReal.mulStrict(bImag))
    .div(denom)
  return tf.complex(real, imag)
}

class ObjectTracker {
  constructor(frame, [xmin, ymin, width, height], debug) {
    this.debug = debug
    this.rect = [xmin, ymin, width, height]

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

    this.gaussFourier = fft2(gaussCrop)

    const processedImageFourier = fft2(processedImage)

    this.Ai = complexMul(this.gaussFourier, conjugate(processedImageFourier))
    this.Bi = complexMul(fft2(imageCrop), conjugate(fft2(imageCrop)))

    // for (let i = 0; i < PRETRAINING_STEPS; i++) {
    //   const processedImage = preprocessImage(randomWarp(imageCrop))
    //   const processedImageFourier = fft2(processedImage)
    //   this.Ai = this.Ai.add(
    //     this.gaussFourier.mulStrict(conjugate(processedImageFourier))
    //   )
    //   this.Bi = this.Bi.add(
    //     processedImageFourier.mulStrict(conjugate(processedImageFourier))
    //   )
    // }

    this.Ai = complexMul(this.Ai, LEARNING_RATE)
    this.Bi = complexMul(this.Bi, LEARNING_RATE)

    this.Ai.print()
    this.Bi.print()
  }

  next = frame => {
    const image = tf.browser.fromPixels(frame)
    const greyscaleImage = rgbToGrayscale(image)

    const [xmin, ymin, width, height] = this.rect
    const imageCrop = greyscaleImage.slice([ymin, xmin], [height, width])
    const processedImage = preprocessImage(imageCrop)

    const Hi = complexDiv(this.Ai, this.Bi)

    const gi = fft2(complexMul(Hi, fft2(processedImage)))

    const normalizedGi = normalize(tf.real(gi))

    tf.browser.toPixels(normalizedGi, this.debug)

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

    console.log(dx)
    console.log(dy)

    // TODO: we need to clip this to bounds.
    this.rect = [Math.round(xmin + dx), Math.round(ymin + dy), width, height]

    const newImageCrop = greyscaleImage.slice(
      [this.rect[1], this.rect[0]],
      [this.rect[3], this.rect[2]]
    )

    const fi = preprocessImage(newImageCrop)

    // fi = greyscaleImage[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
    // fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

    // this.Ai =
    //   LEARNING_RATE * (this.gaussFourier * np.conjugate(np.fft.fft2(fi))) +
    //   (1 - LEARNING_RATE) * this.Ai
    // this.Bi =
    //   LEARNING_RATE * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) +
    //   (1 - LEARNING_RATE) * this.Bi

    const fiFf2 = fft2(fi)
    const aPart1 = complexMul(
      complexMul(this.gaussFourier, conjugate(fiFf2)),
      LEARNING_RATE
    )
    const aPart2 = complexMul(this.Ai, 1 - LEARNING_RATE)

    this.Ai = aPart1.addStrict(aPart2)

    const bPart1 = complexMul(
      complexMul(fiFf2, conjugate(fiFf2)),
      LEARNING_RATE
    )
    const bPart2 = complexMul(this.Bi, 1 - LEARNING_RATE)

    this.Bi = bPart1.addStrict(bPart2)

    return this.rect
  }
}

export default ObjectTracker
