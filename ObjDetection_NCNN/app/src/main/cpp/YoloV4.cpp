#include "YOLOv4.h"

bool YOLOv4::hasGPU = true;
bool YOLOv4::toUseGPU = true;
YOLOv4 *YOLOv4::detector = nullptr;

YOLOv4::YOLOv4(AAssetManager *mgr, const char *param, const char *bin, bool useGPU) {
    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;

    Net = new ncnn::Net();
    // opt 需要在加载前设置
    if (toUseGPU) {
        // enable vulkan compute
        this->Net->opt.use_vulkan_compute = true;
        // turn on for adreno
        this->Net->opt.use_image_storage = true;
        this->Net->opt.use_tensor_storage = true;
    }
    // enable bf16 data type for storage
    // improve most operator performance on all arm devices, may consume more memory
    this->Net->opt.use_bf16_storage = true;

    Net->load_param(mgr, param);
    Net->load_model(mgr, bin);
}

YOLOv4::~YOLOv4() {
    Net->clear();
    delete Net;
}

std::vector<BoxInfo> YOLOv4::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);
    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, input_size,
                                                             input_size);
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);
    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    if (toUseGPU) {
        ex.set_vulkan_compute(toUseGPU);
    }
    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    ncnn::Mat blob;
    ex.extract("output", blob);
    auto boxes = decode_infer(blob, {(int) img_size.width, (int) img_size.height}, input_size, num_class, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());
//    nms(result,nms_threshold);
    return result;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo>
YOLOv4::decode_infer(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold) {
    std::vector<BoxInfo> result;
    for (int i = 0; i < data.h; i++) {
        BoxInfo box;
        const float *values = data.row(i);
        box.label = values[0] - 1;
        box.score = values[1];
//        box.x1 = values[2] * (float) frame_size.width;
//        box.y1 = values[3] * (float) frame_size.height;
//        box.x2 = values[4] * (float) frame_size.width;
//        box.y2 = values[5] * (float) frame_size.height;
        box.x1 = values[2] * (float) frame_size.width;
        box.y1 = values[3] * (float) frame_size.height;
        box.w = values[4] * (float) frame_size.width - values[2] * (float) frame_size.width;
        box.h = values[5] * (float) frame_size.height - values[3] * (float) frame_size.height;
        result.push_back(box);
    }
    return result;
}

