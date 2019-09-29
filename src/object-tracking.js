import * as tf from '@tensorflow/tfjs'
import * as np from './math-util'

const SIGMA = 100
const LEARNING_RATE = 0.125

export default {
  init: (frame, [xmin, ymin, width, height]) => {
    let rect = [xmin, ymin, width, height]

    const image = tf.browser.fromPixels(frame)
    const greyscaleImage = np.rgbToGrayscale(image)

    const center = [ymin + height / 2, xmin + width / 2]
    const gaussTensor = np.gauss([frame.height, frame.width], center, SIGMA)

    const gaussCrop = gaussTensor.slice([ymin, xmin], [height, width])

    const imageCrop = greyscaleImage.slice([ymin, xmin], [height, width])
    const processedImage = np.preprocessImage(imageCrop)

    let gaussFourier = np.dft(gaussCrop)
    const processedImageFourier = np.dft(processedImage)

    let Ai = np.complexMul(gaussFourier, np.conjugate(processedImageFourier))
    let Bi = np.complexMul(np.dft(imageCrop), np.conjugate(np.dft(imageCrop)))

    Ai = np.complexMul(Ai, LEARNING_RATE)
    Bi = np.complexMul(Bi, LEARNING_RATE)

    return {
      track: frame => {
        const image = tf.browser.fromPixels(frame)
        const greyscaleImage = np.rgbToGrayscale(image)

        const [xmin, ymin, width, height] = rect
        const imageCrop = greyscaleImage.slice([ymin, xmin], [height, width])
        const processedImage = np.preprocessImage(imageCrop)

        const Hi = np.complexDiv(Ai, Bi)

        const Gi = np.complexMul(Hi, np.dft(processedImage))
        const gi = np.dft(Gi)

        const [giReal] = gi
        const normalizedGi = np.normalize(giReal)

        const maxValue = tf.max(normalizedGi).dataSync()[0]
        const positions = np.findIndex2d(normalizedGi.arraySync(), maxValue)
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

        // TODO: we need to clip this to bounds.
        rect = [Math.round(xmin - dx), Math.round(ymin - dy), width, height]

        const newImageCrop = greyscaleImage.slice(
          [rect[1], rect[0]],
          [rect[3], rect[2]]
        )

        const fi = np.preprocessImage(newImageCrop)

        const fiFf2 = np.dft(fi)
        const aPart1 = np.complexMul(
          np.complexMul(gaussFourier, np.conjugate(fiFf2)),
          LEARNING_RATE
        )
        const aPart2 = np.complexMul(Ai, 1 - LEARNING_RATE)

        Ai = [aPart1[0].addStrict(aPart2[0]), aPart1[1].addStrict(aPart2[1])]

        const bPart1 = np.complexMul(
          np.complexMul(fiFf2, np.conjugate(fiFf2)),
          LEARNING_RATE
        )
        const bPart2 = np.complexMul(Bi, 1 - LEARNING_RATE)

        Bi = [bPart1[0].addStrict(bPart2[0]), bPart1[1].addStrict(bPart2[1])]

        return rect
      }
    }
  }
}
