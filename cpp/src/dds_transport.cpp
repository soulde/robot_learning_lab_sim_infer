#include "rll_policy/dds_transport.hpp"

#include "rll_policy/detail/dds_messages.hpp"
#include "robot_v1.h"

#include <dds/dds.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace rll_policy {
namespace {
void require_entity(dds_entity_t entity,const std::string& label) {
  if (entity<0) throw std::runtime_error("Cyclone DDS failed to create "+label+" (error "+std::to_string(entity)+")");
}
dds_qos_t* qos(Reliability reliability) {
  auto* value=dds_create_qos();
  if (!value) throw std::runtime_error("Cyclone DDS failed to allocate QoS");
  dds_qset_history(value,DDS_HISTORY_KEEP_LAST,1);
  dds_qset_reliability(value,reliability==Reliability::reliable ? DDS_RELIABILITY_RELIABLE : DDS_RELIABILITY_BEST_EFFORT,
                       DDS_MSECS(100));
  return value;
}
template <typename Sample,typename DomainType,typename Convert>
std::optional<DomainType> take_latest(dds_entity_t reader,Convert convert) {
  std::optional<DomainType> newest;
  constexpr std::size_t batch_size=32;
  std::array<void*,batch_size> samples{};
  std::array<dds_sample_info_t,batch_size> info{};
  while (true) {
    samples.fill(nullptr);
    const dds_return_t count=dds_take(reader,samples.data(),info.data(),samples.size(),samples.size());
    if (count<0) throw std::runtime_error("Cyclone DDS take failed (error "+std::to_string(count)+")");
    if (count==0) break;
    for (dds_return_t i=0;i<count;++i) {
      if (info[static_cast<std::size_t>(i)].valid_data && samples[static_cast<std::size_t>(i)])
        newest=convert(*static_cast<Sample*>(samples[static_cast<std::size_t>(i)]));
    }
    const auto returned=dds_return_loan(reader,samples.data(),static_cast<std::int32_t>(count));
    if (returned<0) throw std::runtime_error("Cyclone DDS failed to return reader loan (error "+std::to_string(returned)+")");
    if (count<static_cast<dds_return_t>(batch_size)) break;
  }
  return newest;
}
void delete_entity(dds_entity_t& entity) noexcept {
  if (entity>=0) dds_delete(entity);
  entity=-1;
}
}

DdsTransport::DdsTransport(const DdsRuntimeConfig& config) {
  try {
    if (!config.cyclonedds_uri.empty() && setenv("CYCLONEDDS_URI",config.cyclonedds_uri.c_str(),1)!=0)
      throw std::runtime_error("failed to set CYCLONEDDS_URI");
    participant_=dds_create_participant(config.domain_id,nullptr,nullptr); require_entity(participant_,"participant");
    state_topic_=dds_create_topic(participant_,&robot_learning_lab_sim_infer_msg_v1_RobotState_desc,config.state_topic.c_str(),nullptr,nullptr); require_entity(state_topic_,"RobotState topic");
    rc_topic_=dds_create_topic(participant_,&robot_learning_lab_sim_infer_msg_v1_RCCommand_desc,config.rc_topic.c_str(),nullptr,nullptr); require_entity(rc_topic_,"RCCommand topic");
    motor_topic_=dds_create_topic(participant_,&robot_learning_lab_sim_infer_msg_v1_MotorCommand_desc,config.motor_command_topic.c_str(),nullptr,nullptr); require_entity(motor_topic_,"MotorCommand topic");
    std::unique_ptr<dds_qos_t,decltype(&dds_delete_qos)> latest(qos(config.sensor_reliability),dds_delete_qos);
    state_reader_=dds_create_reader(participant_,state_topic_,latest.get(),nullptr); require_entity(state_reader_,"RobotState reader");
    rc_reader_=dds_create_reader(participant_,rc_topic_,latest.get(),nullptr); require_entity(rc_reader_,"RCCommand reader");
    std::unique_ptr<dds_qos_t,decltype(&dds_delete_qos)> reliable(qos(config.command_reliability),dds_delete_qos);
    motor_writer_=dds_create_writer(participant_,motor_topic_,reliable.get(),nullptr); require_entity(motor_writer_,"MotorCommand writer");
  } catch (...) { close(); throw; }
}
DdsTransport::~DdsTransport() { close(); }
std::optional<RobotState> DdsTransport::take_latest_state() {
  using Sample=robot_learning_lab_sim_infer_msg_v1_RobotState;
  return take_latest<Sample,RobotState>(state_reader_,[](const Sample& s){return detail::from_idl(s);});
}
std::optional<RCCommand> DdsTransport::take_latest_rc() {
  using Sample=robot_learning_lab_sim_infer_msg_v1_RCCommand;
  return take_latest<Sample,RCCommand>(rc_reader_,[](const Sample& s){return detail::from_idl(s);});
}
void DdsTransport::publish_motor_command(const MotorCommand& command) {
  auto sample=detail::to_idl(command);
  const auto status=dds_write(motor_writer_,&sample);
  robot_learning_lab_sim_infer_msg_v1_MotorCommand_free(&sample,DDS_FREE_CONTENTS);
  if (status<0) throw std::runtime_error("Cyclone DDS MotorCommand write failed (error "+std::to_string(status)+")");
}
void DdsTransport::close() noexcept {
  delete_entity(motor_writer_); delete_entity(state_reader_); delete_entity(rc_reader_);
  delete_entity(motor_topic_); delete_entity(rc_topic_); delete_entity(state_topic_); delete_entity(participant_);
}
}
