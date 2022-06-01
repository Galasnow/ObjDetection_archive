package com.objdetection

import android.content.res.AssetManager
import android.graphics.Bitmap

object YOLOv4 {
    external fun init(manager: AssetManager?, yoloType: Int, useGPU: Boolean)
    external fun detect(bitmap: Bitmap?, threshold: Double, nms_threshold: Double): Array<Box>?

    init {
        System.loadLibrary("objdetection") // 存放在objdetection.so中
    }
}

