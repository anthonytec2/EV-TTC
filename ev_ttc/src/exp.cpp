#include "ev_ttc/config.h"
#include "ev_ttc/exp.h"
#include <chrono>
#include <rclcpp_components/register_node_macro.hpp>
#include <string>
#include <cuda_runtime.h>
#include <fstream>
#include <thread>

using namespace ev_ttc::config; // Use the config namespace for direct access to parameters
using std::placeholders::_1;
using EventPacket = event_camera_msgs::msg::EventPacket;

/**
 * @brief TensorRT logger implementation that suppresses info messages
 Example Inference Engine: https://github.com/cyrusbehr/tensorrt-cpp-api/blob/2e4baa856e876f4700f6bc64f681ace277ab4008/src/engine.h#L61
 */
class Logger : public ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    // Only output warnings and errors
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} logger;

// /**
//  * @brief Check CUDA error code and throw exception if not successful
//  *
//  * @param code CUDA error code
//  Credit to: https://github.com/cyrusbehr/tensorrt-cpp-api/blob/2e4baa856e876f4700f6bc64f681ace277ab4008/include/util/Util.inl#L12
//  */
inline void checkCudaErrorCode(cudaError_t code)
{
  if (code != 0)
  {
    std::string errMsg =
        "CUDA operation failed with code: " + std::to_string(code) + "(" +
        cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
    std::cout << errMsg << std::endl;
    throw std::runtime_error(errMsg);
  }
};

namespace ev_ttc
{
  /**
   * @brief Constructor for the Exp node
   *
   * Sets up ROS2 interfaces, loads TensorRT model, and initializes event processor
   *
   * @param options ROS2 node options
   */
  Exp::Exp(const rclcpp::NodeOptions &options)
      : Node("exp", rclcpp::NodeOptions().use_intra_process_comms(true))
  {
    // Load parameters dynamically
    loadParameters(this);

    // Register dynamic parameter callback
    callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&Exp::paramCallback, this, std::placeholders::_1));

    // ---------- ROS2 Communication Setup ----------
    // QoS profiles for different message types
    auto qos1 = rclcpp::QoS(rclcpp::KeepLast(3)).best_effort().durability_volatile();
    const rmw_qos_profile_t qosProf = rmw_qos_profile_default;
    auto qos2 = rclcpp::QoS(rclcpp::KeepLast(3), rmw_qos_profile_sensor_data);

    // Set up publishers and subscribers
    subscription_ = create_subscription<EventPacket>(
        event_topic, qos1,
        [this](EventPacket::UniquePtr msg)
        { event_message(std::move(msg)); });

    publisherTTC = this->create_publisher<std_msgs::msg::UInt8MultiArray>("/ttc", qos1);
    imagePub_ = image_transport::create_publisher(this, "/image_raw", qosProf);     // Publishes the filtered event image, if imgs
    time_pub = this->create_publisher<std_msgs::msg::Float32>("/time", qos2);       // Publishes the Time taken to process the event packet if Profile
    event_pub = this->create_publisher<std_msgs::msg::Float32>("/event_num", qos2); // Publishes the Number of Events if Profile

    // ---------- Event Processor Setup ----------
    // Create the event decoder for the camera
    decoder = decoderFactory.getInstance("evt3", 1280, 720);
    if (!decoder)
    {
      std::cout << "Invalid encoding: evt3" << std::endl;
      throw std::runtime_error("Invalid encoding!");
    }

    // Set Publisher for Filter
    processor.setPublisher(&imagePub_);

    // Set up thread synchronization
    processor.set_cond(event, event_mtx, work);

    // ---------- TensorRT Model Setup ----------
    // Load the TensorRT engine from file
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile.is_open())
    {
      std::cerr << "Error opening file: " << engine_path << std::endl;
      exit(EXIT_FAILURE);
    }

    // Read the engine file
    engineFile.seekg(0, engineFile.end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> ModelStream(fileSize);
    engineFile.read(ModelStream.data(), fileSize);
    engineFile.close();

    // Create TensorRT runtime and engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(ModelStream.data(), ModelStream.size());
    context = engine->createExecutionContext();

    Dims4 inputDims = {1, num_filters, height, width};
    context->setInputShape("input", inputDims);
    std::cout << "Loaded TensorRT engine" << std::endl;

    // Allocate GPU memory for input and output tensors using constants
    cudaStream_t memAllocStream;
    cudaStreamCreate(&memAllocStream);
    checkCudaErrorCode(cudaMallocAsync(
        &input_buffer_dev, num_filters * height * width * sizeof(__fp16), memAllocStream));
    checkCudaErrorCode(cudaMallocAsync(
        &output_buffer_dev, 1 * height * width * sizeof(__fp16), memAllocStream));
    checkCudaErrorCode(cudaStreamSynchronize(memAllocStream));
    checkCudaErrorCode(cudaStreamDestroy(memAllocStream));

    // Set tensor addresses in TensorRT context
    context->setTensorAddress("input", input_buffer_dev);
    context->setTensorAddress("output", output_buffer_dev);

    // Start inference thread
    modelThread = std::thread(&Exp::inferModel, this);
  }

  /**
   * @brief Destructor for the Exp node
   *
   * Cleans up resources, especially the inference thread
   */
  Exp::~Exp()
  {
    // Signal the inference thread to shut down
    std::unique_lock<std::mutex> lock(event_mtx);
    shutdown = true;
    lock.unlock();
    event.notify_one();

    // Wait for the thread to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    if (modelThread.joinable())
    {
      modelThread.join();
    }

    // Free CUDA memory and TensorRT resources
    if (input_buffer_dev)
      cudaFree(input_buffer_dev);
    if (output_buffer_dev)
      cudaFree(output_buffer_dev);
  }

  // /**
  //  * @brief Callback for dynamic parameter updates
  //  *
  //  * @param params List of parameters to update
  //  * @return Result indicating success or failure
  //  */
  rcl_interfaces::msg::SetParametersResult
  Exp::paramCallback(const std::vector<rclcpp::Parameter> &params)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto &param : params)
    {
      if (param.get_name() == "disp_num")
      {
        RCLCPP_INFO(this->get_logger(), "Updating disp_num: %ld", param.as_int());
        disp_num = param.as_int();
      }
      else if (param.get_name() == "profile")
      {
        RCLCPP_INFO(this->get_logger(), "Updating profile: %s", param.as_bool() ? "true" : "false");
        profile = param.as_bool();
      }
      else if (param.get_name() == "imgs")
      {
        RCLCPP_INFO(this->get_logger(), "Updating imgs: %s", param.as_bool() ? "true" : "false");
        imgs = param.as_bool();
      }
      else if (param.get_name() == "alphas")
      {
        RCLCPP_INFO(this->get_logger(), "Alpha Change");
        alphas = param.as_double_array();
        processor.setFiltConstants();
      }
      else
      {
        result.successful = false;
        result.reason = "Invalid parameter name";
      }
    }

    return result;
  }

  /**
   * @brief Process incoming event packets
   *
   * Decodes events using the event camera codec and passes them to the processor
   *
   * @param msg Event packet message
   */
  void Exp::event_message(EventPacket::UniquePtr msg)
  {

    // Measure decoding time
    auto t1 = std::chrono::high_resolution_clock::now();

    // Process the events
    decoder->decode(&(msg->events[0]), msg->events.size(), &processor); // Process the Event Packet using Callback

    // Publish timing information if profiling is enabled

    if (profile)
    {
      auto t2 = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

      // Publish processing time
      std_msgs::msg::Float32 process_time_msg;
      process_time_msg.data = (float)duration;
      time_pub->publish(process_time_msg);

      // Publish event count
      std_msgs::msg::Float32 event_cnt_msg;
      event_cnt_msg.data = (float)msg->events.size();
      event_pub->publish(event_cnt_msg);
    }
  }

  /**
   * @brief Thread function for neural network inference
   *
   * Waits for processed event data and runs inference using TensorRT
   */
  void Exp::inferModel(void)
  {
    while (true)
    {
      // Wait for work or shutdown signal
      std::unique_lock<std::mutex> lock(event_mtx);
      event.wait(lock, [this]
                 { return shutdown || work; });
      lock.unlock();

      if (shutdown)
      {
        // Exit the thread if shutdown requested
        RCLCPP_INFO(this->get_logger(), "Inference thread shutting down");
        return;
      }
      else if (work)
      {
        // Reset work flag
        lock.lock();
        work = false;
        lock.unlock();
        auto t_now = std::chrono::high_resolution_clock::now();

        // Create CUDA stream for inference
        cudaStream_t inferenceCudaStream;
        cudaStreamCreate(&inferenceCudaStream);

        // Copy input data to GPU using constants
        checkCudaErrorCode(cudaMemcpyAsync(
            input_buffer_dev, processor.exp_filt,
            num_filters * height * width * sizeof(__fp16),
            cudaMemcpyHostToDevice,
            inferenceCudaStream));

        // Run inference
        context->enqueueV3(inferenceCudaStream);

        // Copy output data from GPU
        checkCudaErrorCode(cudaMemcpyAsync(
            ttc, output_buffer_dev,
            height * width * sizeof(__fp16),
            cudaMemcpyDeviceToHost,
            inferenceCudaStream));

        // Create TTC message for publishing
        std_msgs::msg::UInt8MultiArray::UniquePtr msg_ttc = std::make_unique<std_msgs::msg::UInt8MultiArray>();
        msg_ttc->data.resize(height * width * sizeof(__fp16)); // 1 channel, 16-bit data

        // Copy TTC data to message
        checkCudaErrorCode(cudaMemcpyAsync(
            (void *)msg_ttc->data.data(), output_buffer_dev,
            height * width * sizeof(__fp16),
            cudaMemcpyDeviceToHost, inferenceCudaStream));

        // Wait for all operations to complete
        cudaStreamSynchronize(inferenceCudaStream);
        cudaStreamDestroy(inferenceCudaStream);

        // Publish TTC data
        publisherTTC->publish(std::move(msg_ttc));

        // Measure network inference time
        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_now).count();
        RCLCPP_DEBUG(this->get_logger(), "Network Time: %ld μs", duration2);
      }
    }
  }

} // namespace ev_ttc

// Register as a ROS2 component
RCLCPP_COMPONENTS_REGISTER_NODE(ev_ttc::Exp)
