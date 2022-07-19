package com.objdetection

import android.content.res.AssetManager
import android.graphics.Bitmap

object YOLOv4tiny {
    external fun init(manager: AssetManager?, yoloType: Int, useGPU: Boolean)
    external fun detect(
        bitmap: Bitmap?,
        threshold: Float,
        nms_threshold: Float,
        threads_number: Int
    ): Array<Box>?

    init {
        System.loadLibrary("objdetection") // 存放在objdetection.so中
    }
}

