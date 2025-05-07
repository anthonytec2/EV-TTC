#ifndef EV_TTC__EXP_H_
#define EV_TTC__EXP_H_

#include "ev_ttc/config.h" // Include the config:: header
#include "NvInfer.h"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"
#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_msgs/msg/event_packet.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <ev_ttc/ev_processor.h>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>

using namespace nvinfer1;
using namespace ev_ttc::config; // Use the config namespace for direct access to parameters

namespace ev_ttc
{
  /**
   * @brief ROS2 Node for event-based Time-To-Collision estimation
   *
   * This node processes event camera data through multi-scale temporal filters
   * and uses a TensorRT neural network to estimate optical flow.
   */
  class Exp : public rclcpp::Node
  {
  public:
    /**
     * @brief Construct a new Exp node with the given options
     *
     * @param options ROS2 node options
     */
    explicit Exp(const rclcpp::NodeOptions &options);

    /**
     * @brief Clean up resources when node is destroyed
     */
    ~Exp();

  private:
    // ROS2 communication interfaces
    rclcpp::Subscription<EventPacket>::SharedPtr subscription_;                ///< Event Stream subscription
    rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr publisherTTC; ///< TTC output publisher
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr time_pub;             ///< Processing time publisher
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr event_pub;            ///< Event count publisher
    image_transport::Publisher imagePub_;                                      ///< Visualization Exp Filter image publisher
    OnSetParametersCallbackHandle::SharedPtr callback_handle_;                 ///< Parameter callback handler

    /**
     * @brief Process incoming event packets
     *
     * @param msg Event packet message
     */
    void event_message(EventPacket::UniquePtr msg);

    /**
     * @brief Handle dynamic parameter updates
     *
     * @param params List of parameters to update
     * @return Result indicating success or failure
     */
    rcl_interfaces::msg::SetParametersResult
    paramCallback(const std::vector<rclcpp::Parameter> &params);

    /**
     * @brief Thread function for neural network inference
     */
    void inferModel(void);

    // Event processing components
    EVProcessor processor;                                                                             ///< Event data processor
    event_camera_codecs::Decoder<event_camera_codecs::EventPacket, EVProcessor> *decoder;              ///< Event decoder
    event_camera_codecs::DecoderFactory<event_camera_codecs::EventPacket, EVProcessor> decoderFactory; ///< Factory for event decoders

    // TensorRT inference components
    IRuntime *runtime;          ///< TensorRT runtime
    ICudaEngine *engine;        ///< TensorRT engine
    IExecutionContext *context; ///< TensorRT execution context
    void *input_buffer_dev;     ///< Device memory for input tensor
    void *output_buffer_dev;    ///< Device memory for output tensor
    __fp16 ttc[height][width];  ///< Host memory for TTC output
    std::thread modelThread;    ///< Inference thread

    // Thread synchronization
    std::condition_variable event; ///< Condition variable for thread synchronization
    std::mutex event_mtx;          ///< Mutex for thread synchronization model/filter
    bool shutdown{false};          ///< Flag to signal thread shutdown
    bool work{false};              ///< Flag to signal work is available, filter ready for inference
  };
} // namespace ev_ttc
#endif // EV_TTC__EXP_H_