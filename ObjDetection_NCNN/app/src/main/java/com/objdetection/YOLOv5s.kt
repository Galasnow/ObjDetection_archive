package com.objdetection

import android.content.res.AssetManager
import android.graphics.Bitmap

object YOLOv5s {
    external fun init(manager: AssetManager?, useGPU: Boolean)
    external fun detect(
        bitmap: Bitmap?,
        threshold: Float,
        nms_threshold: Float,
        threads_number: Int
    ): Array<Box>?


    init {
        System.loadLibrary("objdetection")
    }
}
