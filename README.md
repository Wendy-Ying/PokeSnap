# PokeSnap

All task are integrated in [camera.py](final/camera.py) and [vision.py](final/vision.py).

## classification
- Make dataset myself, use photo dataset enhancement to expand the dataset, like adjust on brightness, contract, rotation, blur, cutoff, and so on.
- Use opencv to preprocess the images, including channel conversion, hue separation and feature engineering, open and close operations, Otsu method mask extraction, convex packet extraction.
- Use decision tree and adaboost to [predict](adaboost/predict.py) the class of pokemon.
- Accuracy: 86.9%

## beautify
- Split objects and replace backgrounds, deblur with a Wiener filter, add oil painting effect.
- The image can be [beautified](beauty/main.py) by above methods.

## camera
- Use picam to [take photos](camera/camera.py).
- The photos can be saved and processed then.

## printer
- Use [printer](camera/printer_test.py) to print the photo.

## Contributors

The project was a joint effort by Weiyu Chen and me.

<a href="https://github.com/Wendy-Ying">
  <img src="https://avatars.githubusercontent.com/u/143325815?v=4" width="100" />
</a>

<a href="https://github.com/VivianChencwy">
  <img src="https://avatars.githubusercontent.com/u/128114805?v=4"  width="100"/>
</a>
