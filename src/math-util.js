import * as tf from '@tensorflow/tfjs'
tf.enableProdMode()

// NOTE: Don't use tf.complex, it's kinda buggy as of tfjs@1.2.9

const EPSILON = 1e-7

const isComplex = x => !x.dtype && x.length === 2

export const rgbToGrayscale = image =>
  tf.tidy(() => {
    // Reference for converting between RGB and grayscale.
    // https://en.wikipedia.org/wiki/Luma_%28video%29
    const rgbWeights = tf.tensor1d([0.2989, 0.587, 0.114])
    return tf.sum(image.mul(rgbWeights), 2) // broadcast across the image.
  })

export const gauss = ([height, width], [centerY, centerX], sigma) =>
  tf.tidy(() => {
    const x = tf.range(0, width, 1)
    const y = tf.reshape(tf.range(0, height, 1), [-1, 1])

    const dist = tf
      .square(x.sub(centerX))
      .add(tf.square(y.sub(centerY)))
      .div(-2 * sigma)

    return tf.exp(dist)
  })

export const calculateFourierMatrix = ([height, width]) =>
  tf.tidy(() => {
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

    return [[twr, twi], [thr, thi]]
  })

export const dft = (a, t) =>
  tf.tidy(() => {
    let twr
    let twi
    let thr
    let thi
    if (!t) {
      const shape = isComplex(a) ? a[0].shape : a.shape
      ;[[twr, twi], [thr, thi]] = calculateFourierMatrix(shape)
    } else {
      ;[[twr, twi], [thr, thi]] = t
    }

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

export const conjugate = ([real, imag]) => tf.tidy(() => [real, imag.neg()]) // broadcast scalar

export const hanning = M =>
  tf.tidy(() => {
    const numberLine = tf.range(0, M, 1)
    const intermediate = tf.cos(numberLine.mul(2 * Math.PI).div(M - 1)) // multiplying by a scalar
    return tf.scalar(0.5).sub(intermediate.mul(0.5)) // multiplying by a scalar
  })

export const hanningWindow = ([height, width]) =>
  tf.tidy(() => {
    const col = hanning(width)
    const row = hanning(height).reshape([-1, 1])
    return col.mul(row) // broadcast
  })

// NOTE: We could probably cache the hanning window as well.
export const preprocessImage = image =>
  tf.tidy(() => {
    const logOfImage = tf.log1p(image) // log(image + 1)
    const { mean, variance } = tf.moments(logOfImage)
    const normalizedImage = logOfImage
      .sub(mean)
      .div(tf.sqrt(variance).add(EPSILON))

    const window = hanningWindow(image.shape)

    return normalizedImage.mulStrict(window)
  })

export const normalize = tensor =>
  tf.tidy(() =>
    tensor.sub(tf.min(tensor)).div(tf.max(tensor).sub(tf.min(tensor)))
  )

// TODO: Make async.
export const findIndex2d = (matrix, val) => {
  const syncedMatrix = matrix.arraySync()
  const syncedVal = val.dataSync()[0]
  return tf
    .tensor2d(
      syncedMatrix.reduce((acc, row, y) => {
        row.forEach((item, x) => {
          if (item === syncedVal) {
            acc.push([y, x])
          }
        })
        return acc
      }, [])
    )
    .transpose()
}

export const findIndex2dAsync = async (matrix, val) => {
  const positions = await tf.whereAsync(matrix.equal(val))
  return positions.transpose()
}

export const complexMul = (a, b) =>
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

export const complexDiv = ([aReal, aImag], [bReal, bImag]) =>
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
