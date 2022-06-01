package com.objdetection

import android.content.res.AssetManager
import android.graphics.Bitmap

object NanoDet {
    external fun init(manager: AssetManager?, useGPU: Boolean)
    external fun detect(bitmap: Bitmap?, threshold: Double, nms_threshold: Double): Array<Box>?

    init {
        System.loadLibrary("objdetection")
    }
}
