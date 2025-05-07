#ifndef EV_TTC__CONSTANTS_H_
#define EV_TTC__CONSTANTS_H_

#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

namespace ev_ttc
{
    namespace config
    {
        // Image dimensions
        constexpr unsigned int height = 360; ///< Output height
        constexpr unsigned int width = 360;  ///< Output width
        // Filter parameters
        constexpr unsigned int num_filters = 6;                                         ///< Default number of temporal filters
        constexpr float output_time = 7.0f;                                             ///< Output time in milliseconds
        constexpr float dt = 0.2f;                                                      ///< Time bin resolution in milliseconds
        constexpr uint64_t output_time_ns = static_cast<uint64_t>(output_time * 1e6);   ///< Output time in nanoseconds
        constexpr unsigned int time_bins = static_cast<unsigned int>(output_time / dt); ///< Number of time bins
        ///< Output time in nanoseconds (calculated dynamically)
        inline float cx = 602.498762579664f;                                             ///< Camera principal point x-coordinate
        inline float cy = 360.850800481907f;                                             ///< Camera principal point y-coordinate
        inline float fx = 1021.66382499730f;                                             ///< Camera focal length in x
        inline float fy = 1017.70617933675f;                                             ///< Camera focal length in y
        inline float k1 = -0.04940616f;                                                  ///< Radial distortion coefficient k1
        inline float k2 = 0.08712177f;                                                   ///< Radial distortion coefficient k2
        inline float p1 = 0.00039218f;                                                   ///< Tangential distortion coefficient p1
        inline float p2 = -0.00131025f;                                                  ///< Tangential distortion coefficient p2
        inline std::string event_topic = "/event_camera/events";                         ///< Topic for event camera data
        inline std::string engine_path = "/path/to/ttc.engine"; ///< Path to the inference engine
        inline bool profile = true;                                                      ///< Enable profiling
        inline bool imgs = true;                                                         ///< Enable image output
        inline int disp_num = 1;                                                         ///< Display number for debugging
                                                                                         ///< Scale for normalization
        inline std::vector<double> alphas = {0.1, 0.05, 0.025, 0.0125, 0.0075, 0.0035};  ///< Alpha values for filters

        // Load parameters dynamically
        inline void loadParameters(rclcpp::Node *node)
        {
            cx = node->declare_parameter<float>("camera.cx", cx);
            cy = node->declare_parameter<float>("camera.cy", cy);
            fx = node->declare_parameter<float>("camera.fx", fx);
            fy = node->declare_parameter<float>("camera.fy", fy);
            k1 = node->declare_parameter<float>("camera.k1", k1);
            k2 = node->declare_parameter<float>("camera.k2", k2);
            p1 = node->declare_parameter<float>("camera.p1", p1);
            p2 = node->declare_parameter<float>("camera.p2", p2);
            event_topic = node->declare_parameter<std::string>("event_topic", event_topic);
            profile = node->declare_parameter<bool>("profile", profile);
            imgs = node->declare_parameter<bool>("imgs", imgs);
            disp_num = node->declare_parameter<int>("disp_num", disp_num);
            alphas = node->declare_parameter<std::vector<double>>("alphas", alphas);
            engine_path = node->declare_parameter<std::string>("engine_path", engine_path);
        }
    } // namespace config
} // namespace ev_ttc

#endif // EV_TTC__CONSTANTS_H_