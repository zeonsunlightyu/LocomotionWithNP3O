/****************************************************************

  Generated by Eclipse Cyclone DDS IDL to CXX Translator
  File name: Req_.idl
  Source: Req_.hpp
  Cyclone DDS: v0.10.2

*****************************************************************/
#ifndef DDSCXX_REQ__HPP
#define DDSCXX_REQ__HPP

#include <string>

namespace unitree_go
{
namespace msg
{
namespace dds_
{
class Req_
{
private:
 std::string uuid_;
 std::string body_;

public:
  Req_() = default;

  explicit Req_(
    const std::string& uuid,
    const std::string& body) :
    uuid_(uuid),
    body_(body) { }

  const std::string& uuid() const { return this->uuid_; }
  std::string& uuid() { return this->uuid_; }
  void uuid(const std::string& _val_) { this->uuid_ = _val_; }
  void uuid(std::string&& _val_) { this->uuid_ = _val_; }
  const std::string& body() const { return this->body_; }
  std::string& body() { return this->body_; }
  void body(const std::string& _val_) { this->body_ = _val_; }
  void body(std::string&& _val_) { this->body_ = _val_; }

  bool operator==(const Req_& _other) const
  {
    (void) _other;
    return uuid_ == _other.uuid_ &&
      body_ == _other.body_;
  }

  bool operator!=(const Req_& _other) const
  {
    return !(*this == _other);
  }

};

}

}

}

#include "dds/topic/TopicTraits.hpp"
#include "org/eclipse/cyclonedds/topic/datatopic.hpp"

namespace org {
namespace eclipse {
namespace cyclonedds {
namespace topic {

template <> constexpr const char* TopicTraits<::unitree_go::msg::dds_::Req_>::getTypeName()
{
  return "unitree_go::msg::dds_::Req_";
}

template <> constexpr bool TopicTraits<::unitree_go::msg::dds_::Req_>::isSelfContained()
{
  return false;
}

template <> constexpr bool TopicTraits<::unitree_go::msg::dds_::Req_>::isKeyless()
{
  return true;
}

#ifdef DDSCXX_HAS_TYPE_DISCOVERY
template<> constexpr unsigned int TopicTraits<::unitree_go::msg::dds_::Req_>::type_map_blob_sz() { return 246; }
template<> constexpr unsigned int TopicTraits<::unitree_go::msg::dds_::Req_>::type_info_blob_sz() { return 100; }
template<> inline const uint8_t * TopicTraits<::unitree_go::msg::dds_::Req_>::type_map_blob() {
  static const uint8_t blob[] = {
 0x4c,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0xf1,  0x65,  0xcc,  0xb2,  0xec,  0x1b,  0x91,  0xab, 
 0x31,  0x5f,  0x50,  0x70,  0x86,  0x74,  0x52,  0x00,  0x34,  0x00,  0x00,  0x00,  0xf1,  0x51,  0x01,  0x00, 
 0x01,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x24,  0x00,  0x00,  0x00,  0x02,  0x00,  0x00,  0x00, 
 0x0c,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x01,  0x00,  0x70,  0x00,  0xef,  0x7c,  0x87,  0x6f, 
 0x0c,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0x01,  0x00,  0x70,  0x00,  0x84,  0x1a,  0x2d,  0x68, 
 0x7b,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0xf2,  0x03,  0x02,  0x36,  0x18,  0x03,  0xef,  0x61, 
 0x1d,  0xd6,  0x73,  0x0b,  0x4b,  0x4d,  0x57,  0x00,  0x63,  0x00,  0x00,  0x00,  0xf2,  0x51,  0x01,  0x00, 
 0x24,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x1c,  0x00,  0x00,  0x00,  0x75,  0x6e,  0x69,  0x74, 
 0x72,  0x65,  0x65,  0x5f,  0x67,  0x6f,  0x3a,  0x3a,  0x6d,  0x73,  0x67,  0x3a,  0x3a,  0x64,  0x64,  0x73, 
 0x5f,  0x3a,  0x3a,  0x52,  0x65,  0x71,  0x5f,  0x00,  0x33,  0x00,  0x00,  0x00,  0x02,  0x00,  0x00,  0x00, 
 0x13,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x01,  0x00,  0x70,  0x00,  0x05,  0x00,  0x00,  0x00, 
 0x75,  0x75,  0x69,  0x64,  0x00,  0x00,  0x00,  0x00,  0x13,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00, 
 0x01,  0x00,  0x70,  0x00,  0x05,  0x00,  0x00,  0x00,  0x62,  0x6f,  0x64,  0x79,  0x00,  0x00,  0x00,  0x00, 
 0x22,  0x00,  0x00,  0x00,  0x01,  0x00,  0x00,  0x00,  0xf2,  0x03,  0x02,  0x36,  0x18,  0x03,  0xef,  0x61, 
 0x1d,  0xd6,  0x73,  0x0b,  0x4b,  0x4d,  0x57,  0xf1,  0x65,  0xcc,  0xb2,  0xec,  0x1b,  0x91,  0xab,  0x31, 
 0x5f,  0x50,  0x70,  0x86,  0x74,  0x52, };
  return blob;
}
template<> inline const uint8_t * TopicTraits<::unitree_go::msg::dds_::Req_>::type_info_blob() {
  static const uint8_t blob[] = {
 0x60,  0x00,  0x00,  0x00,  0x01,  0x10,  0x00,  0x40,  0x28,  0x00,  0x00,  0x00,  0x24,  0x00,  0x00,  0x00, 
 0x14,  0x00,  0x00,  0x00,  0xf1,  0x65,  0xcc,  0xb2,  0xec,  0x1b,  0x91,  0xab,  0x31,  0x5f,  0x50,  0x70, 
 0x86,  0x74,  0x52,  0x00,  0x38,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x04,  0x00,  0x00,  0x00, 
 0x00,  0x00,  0x00,  0x00,  0x02,  0x10,  0x00,  0x40,  0x28,  0x00,  0x00,  0x00,  0x24,  0x00,  0x00,  0x00, 
 0x14,  0x00,  0x00,  0x00,  0xf2,  0x03,  0x02,  0x36,  0x18,  0x03,  0xef,  0x61,  0x1d,  0xd6,  0x73,  0x0b, 
 0x4b,  0x4d,  0x57,  0x00,  0x67,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x00,  0x04,  0x00,  0x00,  0x00, 
 0x00,  0x00,  0x00,  0x00, };
  return blob;
}
#endif //DDSCXX_HAS_TYPE_DISCOVERY

} //namespace topic
} //namespace cyclonedds
} //namespace eclipse
} //namespace org

namespace dds {
namespace topic {

template <>
struct topic_type_name<::unitree_go::msg::dds_::Req_>
{
    static std::string value()
    {
      return org::eclipse::cyclonedds::topic::TopicTraits<::unitree_go::msg::dds_::Req_>::getTypeName();
    }
};

}
}

REGISTER_TOPIC_TYPE(::unitree_go::msg::dds_::Req_)

namespace org{
namespace eclipse{
namespace cyclonedds{
namespace core{
namespace cdr{

template<>
propvec &get_type_props<::unitree_go::msg::dds_::Req_>();

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool write(T& streamer, const ::unitree_go::msg::dds_::Req_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!write_string(streamer, instance.uuid(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!write_string(streamer, instance.body(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool write(S& str, const ::unitree_go::msg::dds_::Req_& instance, bool as_key) {
  auto &props = get_type_props<::unitree_go::msg::dds_::Req_>();
  str.set_mode(cdr_stream::stream_mode::write, as_key);
  return write(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool read(T& streamer, ::unitree_go::msg::dds_::Req_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!read_string(streamer, instance.uuid(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!read_string(streamer, instance.body(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool read(S& str, ::unitree_go::msg::dds_::Req_& instance, bool as_key) {
  auto &props = get_type_props<::unitree_go::msg::dds_::Req_>();
  str.set_mode(cdr_stream::stream_mode::read, as_key);
  return read(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool move(T& streamer, const ::unitree_go::msg::dds_::Req_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!move_string(streamer, instance.uuid(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!move_string(streamer, instance.body(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool move(S& str, const ::unitree_go::msg::dds_::Req_& instance, bool as_key) {
  auto &props = get_type_props<::unitree_go::msg::dds_::Req_>();
  str.set_mode(cdr_stream::stream_mode::move, as_key);
  return move(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool max(T& streamer, const ::unitree_go::msg::dds_::Req_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!max_string(streamer, instance.uuid(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
      case 1:
      if (!streamer.start_member(*prop))
        return false;
      if (!max_string(streamer, instance.body(), 0))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool max(S& str, const ::unitree_go::msg::dds_::Req_& instance, bool as_key) {
  auto &props = get_type_props<::unitree_go::msg::dds_::Req_>();
  str.set_mode(cdr_stream::stream_mode::max, as_key);
  return max(str, instance, props.data()); 
}

} //namespace cdr
} //namespace core
} //namespace cyclonedds
} //namespace eclipse
} //namespace org

#endif // DDSCXX_REQ__HPP