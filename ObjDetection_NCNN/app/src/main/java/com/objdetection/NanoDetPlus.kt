package com.objdetection

import android.content.res.AssetManager
import android.graphics.Bitmap

object NanoDetPlus {
    external fun init(manager: AssetManager?, useGPU: Boolean)
    external fun detect(
        bitmap: Bitmap?,
        threshold: Float,
        nms_threshold: Float,
        threadsNumber: Int
    ): Array<Box>?

    init {
        System.loadLibrary("objdetection")
    }
}
