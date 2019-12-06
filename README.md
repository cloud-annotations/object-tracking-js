# Object Tracking JavaScript SDK
[![NPM Version](https://img.shields.io/npm/v/@cloud-annotations/object-tracking.svg)](https://npmjs.org/package/@cloud-annotations/object-tracking)
[![NPM Downloads](https://img.shields.io/npm/dm/@cloud-annotations/object-tracking.svg)](https://npmjs.org/package/@cloud-annotations/object-tracking)

Simple object tracking in js
![Demo](https://media.giphy.com/media/YpvymI4gUGVHhSaT5C/giphy.gif)

## Installation
```bash
npm install @cloud-annotations/object-tracking
```

## Usage
```js
import objectTracker from '@cloud-annotations/object-tracking'

const frame1 = document.getElementById('img1')
const frame2 = document.getElementById('img2')
const frame3 = document.getElementById('img3')
//    ...
const frameN = document.getElementById('imgN')

const tracker = objectTracker.init(frame1, [x, y, width, height])
const box2 = await tracker.next(frame2)
const box3 = await tracker.next(frame3)
//    ...
const boxN = await tracker.next(frameN)

// box =>
[x, y, width, height]
```

## Usage via Script Tag
```html
<script src="https://cdn.jsdelivr.net/npm/@cloud-annotations/object-tracking"></script>
```
