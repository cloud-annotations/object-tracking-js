import * as tf from '@tensorflow/tfjs'

// NOTE: Don't use tf.complex, it's super buggy as of tfjs@1.2.9

const EPSILON = 1e-7
const SIGMA = 100
const LEARNING_RATE = 0.125

const isComplex = x => !x.dtype && x.length === 2

const rgbToGrayscale = image =>
  tf.tidy(() => {
    // Reference for converting between RGB and grayscale.
    // https://en.wikipedia.org/wiki/Luma_%28video%29
    const rgbWeights = tf.tensor1d([0.2989, 0.587, 0.114])
    return tf.sum(image.mul(rgbWeights), 2) // broadcast across the image.
  })

const gauss = ([height, width], [centerY, centerX], sigma) =>
  tf.tidy(() => {
    const x = tf.range(0, width, 1)
    const y = tf.reshape(tf.range(0, height, 1), [-1, 1])

    const dist = tf
      .square(x.sub(centerX))
      .add(tf.square(y.sub(centerY)))
      .div(-2 * sigma)

    return tf.exp(dist)
  })

const dft = a =>
  tf.tidy(() => {
    const [height, width] = isComplex(a) ? a[0].shape : a.shape

    const widthNumberLine = tf.range(0, width, 1)
    const xyw = widthNumberLine.mul(tf.reshape(widthNumberLine, [-1, 1]))

    const heightNumberLine = tf.range(0, height, 1)
    const xyh = heightNumberLine.mul(tf.reshape(heightNumberLine, [-1, 1]))

    // using euler's formula e^(-2 * pi * i / N) =>
    // real:  cos(2 * pi / N)
    // imag: -sin(2 * pi / N)

    const realWW = Math.cos((2 * Math.PI) / width)
    const imagWW = -Math.sin((2 * Math.PI) / width)

    const realWH = Math.cos((2 * Math.PI) / height)
    const imagWH = -Math.sin((2 * Math.PI) / height)

    // (r + i)^N =>
    // real: (r^2 + i^2)^N * cos(N * atan(i / r))
    // imag: (r^2 + i^2)^N * sin(N * atan(i / r))

    const pow1 = tf.pow(tf.square(realWW).add(tf.square(imagWW)), xyw)
    const inner1 = tf.atan(tf.div(imagWW, realWW)).mul(xyw)

    const twr = pow1.mul(tf.cos(inner1))
    const twi = pow1.mul(tf.sin(inner1))

    const pow2 = tf.pow(tf.square(realWH).add(tf.square(imagWH)), xyh)
    const inner2 = tf.atan(tf.div(imagWH, realWH)).mul(xyh)

    const thr = pow2.mul(tf.cos(inner2))
    const thi = pow2.mul(tf.sin(inner2))

    // num * complex == num * real + num * imag)
    const gt = (() => {
      if (isComplex(a)) {
        const [aReal, aImag] = a
        const r1r2 = tf.matMul(aReal, twr)
        const r1i2 = tf.matMul(aReal, twi)
        const i1r2 = tf.matMul(aImag, twr)
        const i1i2 = tf.matMul(aImag, twi)

        const real = r1r2.sub(i1i2)
        const imag = r1i2.add(i1r2)

        return [real, imag]
      }
      return [tf.matMul(a, twr), tf.matMul(a, twi)]
    })()

    // complex * complex == real * real + real * imag + imag * real + imag * imag)
    const [gtr, gti] = gt
    const r1r2 = tf.transpose(tf.matMul(gtr, thr, true))
    const r1i2 = tf.transpose(tf.matMul(gtr, thi, true))
    const i1r2 = tf.transpose(tf.matMul(gti, thr, true))
    const i1i2 = tf.transpose(tf.matMul(gti, thi, true))

    const real = r1r2.sub(i1i2)
    const imag = r1i2.add(i1r2)

    return [real, imag]
  })

const conjugate = ([real, imag]) => tf.tidy(() => [real, imag.neg()]) // broadcast scalar

const hanning = M =>
  tf.tidy(() => {
    const numberLine = tf.linspace(0, M - 1, M)
    const intermediate = tf.cos(numberLine.mul(2 * Math.PI).div(M - 1)) // multiplying by a scalar
    return tf.scalar(0.5).sub(intermediate.mul(0.5)) // multiplying by a scalar
  })

const hanningWindow = (width, height) =>
  tf.tidy(() => {
    const col = hanning(width)
    const row = hanning(height).reshape([-1, 1])
    return col.mul(row) // broadcast
  })

const preprocessImage = image =>
  tf.tidy(() => {
    const [height, width] = image.shape
    const logOfImage = tf.log1p(image) // log(image + 1)
    const stdOfImage = tf.moments(logOfImage).variance.sqrt()
    const normalizedImage = logOfImage
      .sub(tf.mean(logOfImage))
      .div(stdOfImage.add(EPSILON))

    const window = hanningWindow(width, height)

    return normalizedImage.mulStrict(window)
  })

const normalize = tensor =>
  tf.tidy(() =>
    tensor.sub(tf.min(tensor)).div(tf.max(tensor).sub(tf.min(tensor)))
  )

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

const complexMul = (a, b) =>
  tf.tidy(() => {
    // CASES:
    // complex * complex
    if (isComplex(a) && isComplex(b)) {
      const [aReal, aImag] = a
      const [bReal, bImag] = b
      const r1r2 = aReal.mulStrict(bReal)
      const r1i2 = aReal.mulStrict(bImag)
      const i1r2 = aImag.mulStrict(bReal)
      const i1i2 = aImag.mulStrict(bImag)

      const real = r1r2.sub(i1i2)
      const imag = r1i2.add(i1r2)
      return [real, imag]
    }

    // complex * tensor
    // complex * scalar
    if (isComplex(a)) {
      const [aReal, aImag] = a
      return [aReal.mul(b), aImag.mul(b)] // allow broadcast
    }

    // tensor * complex
    // scalar * complex
    if (isComplex(b)) {
      const [bReal, bImag] = b
      return [bReal.mul(a), bImag.mul(a)] // allow broadcast
    }

    // tensor * tensor
    // tensor * scalar
    // scalar * tensor
    // scalar * scalar
    return tf.mul(a, b) // allow broadcast
  })

const complexDiv = ([aReal, aImag], [bReal, bImag]) =>
  tf.tidy(() => {
    const denom = bImag.mulStrict(bImag).add(bReal.mulStrict(bReal))
    const real = aReal
      .mulStrict(bReal)
      .add(aImag.mulStrict(bImag))
      .div(denom)
    const imag = bReal
      .mulStrict(aImag)
      .sub(aReal.mulStrict(bImag))
      .div(denom)
    return [real, imag]
  })

class ObjectTracker {
  constructor(frame, [xmin, ymin, width, height], debug) {
    this.debug = debug
    this.rect = [xmin, ymin, width, height]

    const image = tf.browser.fromPixels(frame)
    const greyscaleImage = rgbToGrayscale(image)

    const center = [ymin + height / 2, xmin + width / 2]
    const gaussTensor = gauss([frame.height, frame.width], center, SIGMA)

    const gaussCrop = gaussTensor.slice([ymin, xmin], [height, width])

    const imageCrop = greyscaleImage.slice([ymin, xmin], [height, width])
    const processedImage = preprocessImage(imageCrop)

    // // DEBUG //
    // const fakeResponse = tf.tensor2d([
    //   [0, 0, 0, 0, 0, 0],
    //   [0, 0, 0, 0, 0, 0],
    //   [0, 0, 1, 1, 0, 0],
    //   [0, 0, 1, 1, 0, 0],
    //   [0, 0, 1, 1, 0, 0],
    //   [0, 0, 0, 0, 0, 0],
    //   [0, 0, 0, 0, 0, 0],
    //   [0, 0, 0, 0, 0, 0],
    //   [0, 0, 0, 0, 0, 0],
    //   [0, 0, 0, 0, 0, 0],
    //   [0, 0, 0, 0, 0, 0]
    // ])

    // const G1 = dft(fakeResponse)
    // G1[0].print()
    // G1[1].print()
    // const G2 = dft(G1)
    // G2[0].print()
    // G2[1].print()
    // DEBUG //

    // tf.browser.toPixels(fakeResponse, this.debug)

    this.gaussFourier = dft(gaussCrop)
    const processedImageFourier = dft(processedImage)

    this.Ai = complexMul(this.gaussFourier, conjugate(processedImageFourier))
    this.Bi = complexMul(dft(imageCrop), conjugate(dft(imageCrop)))

    this.Ai = complexMul(this.Ai, LEARNING_RATE)
    this.Bi = complexMul(this.Bi, LEARNING_RATE)
  }

  next = frame => {
    const image = tf.browser.fromPixels(frame)
    const greyscaleImage = rgbToGrayscale(image)

    const [xmin, ymin, width, height] = this.rect
    const imageCrop = greyscaleImage.slice([ymin, xmin], [height, width])
    const processedImage = preprocessImage(imageCrop)

    const Hi = complexDiv(this.Ai, this.Bi)

    const Gi = complexMul(Hi, dft(processedImage))
    const gi = dft(Gi)

    // gi[0].print()
    // gi[1].print()

    const [giReal] = gi
    const normalizedGi = normalize(giReal)

    // tf.browser.toPixels(normalizedGi, this.debug)

    const maxValue = tf.max(normalizedGi).dataSync()[0]
    const positions = findIndex2d(normalizedGi.arraySync(), maxValue)
    const positionsTransposed = tf.tensor2d(positions).transpose()

    const dy = tf
      .mean(positionsTransposed.slice(0, 1))
      .sub(normalizedGi.shape[0] / 2)
      .round()
      .dataSync()[0]
    const dx = tf
      .mean(positionsTransposed.slice(1))
      .sub(normalizedGi.shape[1] / 2)
      .round()
      .dataSync()[0]

    console.log(dx)
    console.log(dy)

    // TODO: we need to clip this to bounds.
    this.rect = [Math.round(xmin - dx), Math.round(ymin - dy), width, height]

    const newImageCrop = greyscaleImage.slice(
      [this.rect[1], this.rect[0]],
      [this.rect[3], this.rect[2]]
    )

    tf.browser.toPixels(newImageCrop.div(255), this.debug)

    const fi = preprocessImage(newImageCrop)

    const fiFf2 = dft(fi)
    const aPart1 = complexMul(
      complexMul(this.gaussFourier, conjugate(fiFf2)),
      LEARNING_RATE
    )
    const aPart2 = complexMul(this.Ai, 1 - LEARNING_RATE)

    this.Ai = [aPart1[0].addStrict(aPart2[0]), aPart1[1].addStrict(aPart2[1])]

    const bPart1 = complexMul(
      complexMul(fiFf2, conjugate(fiFf2)),
      LEARNING_RATE
    )
    const bPart2 = complexMul(this.Bi, 1 - LEARNING_RATE)

    this.Bi = [bPart1[0].addStrict(bPart2[0]), bPart1[1].addStrict(bPart2[1])]

    return this.rect
  }
}

export default ObjectTracker
