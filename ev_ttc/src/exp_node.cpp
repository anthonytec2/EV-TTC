/**
 * @file exp_node.cpp
 * @brief Entry point for the EV-TTC ROS2 node
 *
 * This file provides the main function for running the Exp node as a standalone process.
 */
 #include <memory>
 #include <rclcpp/rclcpp.hpp>
 #include "ev_ttc/exp.h"
 
 int main(int argc, char **argv)
 {
   // Initialize ROS2
   rclcpp::init(argc, argv);
 
   // Create the Exp node
   auto node = std::make_shared<ev_ttc::Exp>(rclcpp::NodeOptions());
 
   RCLCPP_INFO(node->get_logger(), "EV-TTC node started");
 
   // Run the node
   rclcpp::spin(node);
 
   // Clean up when done
   rclcpp::shutdown();
   return 0;
 }