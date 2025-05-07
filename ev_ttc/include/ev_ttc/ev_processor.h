#ifndef EV_TTC__EV_CACHE_H_
#define EV_TTC__EV_CACHE_H_

#include "ev_ttc/config.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <event_camera_codecs/event_processor.h>
#include <sensor_msgs/msg/image.hpp>
#include "image_transport/image_transport.hpp"
#include <fstream>
#include <cassert>

using event_camera_codecs::EventPacket;
using namespace ev_ttc::config; // Use the config namespace for direct access to parameters

namespace ev_ttc
{
    /**
     * @brief Processes event camera data through multi-scale exponential filters
     *
     * This class implements the event_camera_codecs::EventProcessor interface,
     * applying multiple temporal exponential filters to incoming events. The filtered
     * data is used as input to a neural network for TTC estimation.
     */
    class EVProcessor : public event_camera_codecs::EventProcessor
    {
    public:
        uint64_t last_active = 0;                          ///< Timestamp of last processed event
        __fp16 exp_filt[num_filters][height][width];       ///< Filtered event data
        __fp16 filt_constants[num_filters][time_bins + 1]; ///< Pre-computed filter constants

        // Communication with neural network thread
        void *input_buffer_dev_ev;      ///< Device memory for network input
        std::condition_variable *event; ///< Condition variable for thread synchronization
        std::mutex *event_mtx;          ///< Mutex for thread synchronization
        bool *work;                     ///< Flag to signal work is available

        image_transport::Publisher *imagePub_; ///< Image publisher

        /**
         * @brief Set the publisher for filter visualization
         *
         * @param pub2 Image transport publisher
         */
        void setPublisher(image_transport::Publisher *pub2)
        {
            imagePub_ = pub2;
            setFiltConstants();
        }

        /**
         * @brief Set synchronization primitives for inter-thread communication
         *
         * @param ev Condition variable
         * @param ev_mtx Mutex
         * @param workTmp Work flag
         */
        void set_cond(std::condition_variable &ev, std::mutex &ev_mtx, bool &workTmp)
        {
            work = &workTmp;
            event = &ev;
            event_mtx = &ev_mtx;
        }

        /**
         * @brief Pre-compute filter constants for all time bins
         */
        void setFiltConstants()
        {
            for (unsigned int i = 0; i < num_filters; i++)
            {
                for (unsigned int j = 0; j < time_bins; j++)
                {
                    filt_constants[i][j] = alphas[i] * std::pow(1 - alphas[i], -((float)j));
                }
                filt_constants[i][time_bins] = std::pow(1 - alphas[i], (float)time_bins);
            }
        }

        /**
         * @brief Reset all filter values to zero
         */
        void clearFilter()
        {
            for (int i = 0; i < num_filters; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    for (int k = 0; k < width; k++)
                    {
                        exp_filt[i][j][k] = 0;
                    }
                }
            }
        }

        /**
         * @brief Process a change detection event
         *
         * This is called by the event decoder for each event in the stream.
         * The function applies the event to multiple exponential filters and
         * triggers neural network inference when enough time has passed.
         *
         * @param time Event timestamp in nanoseconds
         * @param ex Event x-coordinate (from raw sensor)
         * @param ey Event y-coordinate (from raw sensor)
         * @param polarity Event polarity (1=positive, 0=negative)
         */
        void eventCD(uint64_t time, uint16_t ex, uint16_t ey, uint8_t polarity) override
        {

            // For first event, initialize timestamp
            if (last_active == 0)
            {
                last_active = time;
            }

            // Check if enough time has passed to produce output
            uint64_t delta_time = (time - last_active);
            if (delta_time >= output_time_ns)
            {
                // Apply exponential decay to all filters
                for (int i = 0; i < num_filters; i++)
                {
                    float filt_const_i = filt_constants[i][time_bins];
                    for (int j = 0; j < height; j++)
                    {
                        for (int k = 0; k < width; k++)
                        {
                            exp_filt[i][j][k] = exp_filt[i][j][k] * filt_const_i;
                        }
                    }
                }

                // Copy filtered data to GPU and signal inference thread
                cudaMemcpy(input_buffer_dev_ev, exp_filt,
                           num_filters * height * width * sizeof(__fp16),
                           cudaMemcpyHostToDevice);
                std::unique_lock<std::mutex> lock(*event_mtx);
                *work = true;
                lock.unlock();
                event->notify_one();

                // Publish visualization if enabled
                publishVisualization();

                // Update timestamp for next output
                last_active = time;
                delta_time = 0;
            }

            // Convert time delta to bin index
            unsigned int act_ind = delta_time * (1e-6 * (1 / dt)); // Convert Delta T to discrete bin
            assert((act_ind < time_bins) && "Index exceeds total time bins");

            // Apply lens distortion correction
            auto [x_und, y_und] = undistortEvent(
                ex, ey, cx, cy, fx, fy, k1, k2, p1, p2);

            const float ds_x = x_und / 2 - 140;
            const float ds_y = y_und / 2;

            // Skip events outside the region of interest
            if (ds_x < 0 || ds_y < 0 || ds_x > 358 || ds_y > 358)
            {

                return;
            }

            // Bilinear interpolation for smoother event placement
            uint16_t nc1 = ds_x;
            uint16_t nc2 = ds_y;
            __fp16 e1 = (1 - (ds_x - nc1)) * (1 - (ds_y - nc2));
            __fp16 e2 = (1 - (nc1 + 1 - ds_x)) * (1 - (ds_y - nc2));
            __fp16 e3 = (1 - (ds_x - nc1)) * (1 - (nc2 + 1 - ds_y));
            __fp16 e4 = (1 - (nc1 + 1 - ds_x)) * (1 - (nc2 + 1 - ds_y));
            const float sign = (polarity) ? 1 : -1;

            // Apply event to all filters with appropriate time constant
            for (int i = 0; i < num_filters; i++)
            {
                exp_filt[i][nc2][nc1] += filt_constants[i][act_ind] * sign * e1;
                exp_filt[i][nc2][nc1 + 1] += filt_constants[i][act_ind] * sign * e2;
                exp_filt[i][nc2 + 1][nc1] += filt_constants[i][act_ind] * sign * e3;
                exp_filt[i][nc2 + 1][nc1 + 1] += filt_constants[i][act_ind] * sign * e4;
            }
        }

        /**
         * @brief Publish the filtered data as a visualization image
         *
         * This function creates an image message from the filtered data and publishes
         * it using the configured image transport publisher. The image represents the
         * output of the selected filter, then visuazlized.
         */
        void publishVisualization()
        {
            if (!imgs)
                return;

            // Create and configure the image message
            auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
            image_msg->header.frame_id = "camera_frame";
            image_msg->encoding = "mono8";
            image_msg->width = width;
            image_msg->height = height;
            image_msg->step = width;
            image_msg->data.resize(width * height);

            // Find the min and max values in the filter for normalization
            __fp16 min_val = std::numeric_limits<__fp16>::max();
            __fp16 max_val = std::numeric_limits<__fp16>::lowest();

            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    auto val = exp_filt[disp_num][j][k];
                    if (val < min_val)
                        min_val = val;
                    if (val > max_val)
                        max_val = val;
                }
            }

            // Normalize and convert to uint8 for visualization
            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    auto val = exp_filt[disp_num][j][k];
                    uint8_t normalized_val = static_cast<uint8_t>(
                        255 * (val - min_val) / (max_val - min_val));
                    image_msg->data[j * width + k] = normalized_val;
                }
            }

            // Publish the image
            imagePub_->publish(std::move(image_msg));
        }

        /**
         * @brief Apply lens distortion correction to an event's coordinates
         *
         * @param ex Event x-coordinate (raw sensor)
         * @param ey Event y-coordinate (raw sensor)
         * @param cx Principal point x-coordinate
         * @param cy Principal point y-coordinate
         * @param fx Focal length x
         * @param fy Focal length y
         * @param k1 Radial distortion coefficient k1
         * @param k2 Radial distortion coefficient k2
         * @param p1 Tangential distortion coefficient p1
         * @param p2 Tangential distortion coefficient p2
         * @return std::pair<float, float> Corrected x and y coordinates
         */
        std::pair<float, float> undistortEvent(
            float ex, float ey,
            float cx, float cy,
            float fx, float fy,
            float k1, float k2,
            float p1, float p2)
        {
            // CV2 Undistort was way slower than using the bellow
            //  Normalize coordinates
            float cx_norm = (ex - cx) / fx;
            float cy_norm = (ey - cy) / fy;

            // Compute radial distortion
            float r = std::pow(cx_norm, 2) + std::pow(cy_norm, 2);
            float t1 = 1 + k1 * r + k2 * std::pow(r, 2);

            // Compute tangential distortion
            float crossterm = cx_norm * cy_norm;
            float x_pp = (cx_norm * t1 + 2 * p1 * crossterm + p2 * (r + 2 * std::pow(cx_norm, 2)));
            float y_pp = (cy_norm * t1 + 2 * p2 * crossterm + p1 * (r + 2 * std::pow(cy_norm, 2)));

            // Convert back to pixel coordinates
            float x_pixel = x_pp * fx + cx;
            float y_pixel = y_pp * fy + cy;

            return {x_pixel, y_pixel};
        }

        // Unused interface methods
        void eventExtTrigger(uint64_t sensor_time, uint8_t edge, uint8_t id) override {}
        void finished() override {}
        void rawData(const char *, size_t) override {}
    };
} // namespace ev_ttc

#endif // EV_TTC__EV_CACHE_H_