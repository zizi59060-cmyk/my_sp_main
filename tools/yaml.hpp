// #ifndef TOOLS__YAML_HPP
// #define TOOLS__YAML_HPP

// #include <yaml-cpp/yaml.h>

// #include "tools/logger.hpp"

// namespace tools
// {
// inline YAML::Node load(const std::string & path)
// {
//   try {
//     return YAML::LoadFile(path);
//   } catch (const YAML::BadFile & e) {
//     logger()->error("[YAML] Failed to load file: {}", e.what());
//     exit(1);
//   } catch (const YAML::ParserException & e) {
//     logger()->error("[YAML] Parser error: {}", e.what());
//     exit(1);
//   }
// }

// template <typename T>
// inline T read(const YAML::Node & yaml, const std::string & key)
// {
//   if (yaml[key]) return yaml[key].as<T>();
//   logger()->error("[YAML] {} not found!", key);
//   exit(1);
// }

// }  // namespace tools

// #endif  // TOOLS__YAML_HPP

#pragma once
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include "logger.hpp"

namespace tools
{

// 全局加载函数：安全加载 YAML 文件
inline YAML::Node load(const std::string &path)
{
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::BadFile &e) {
    logger()->error("[YAML] 无法打开配置文件: {} ({})", path, e.what());
    throw std::runtime_error("YAML file not found: " + path);
  } catch (const std::exception &e) {
    logger()->error("[YAML] 加载文件 {} 时出错: {}", path, e.what());
    throw;
  }
}

// 模板读取函数：带类型与存在性检查
template <typename T>
inline T read(const YAML::Node &yaml, const std::string &key)
{
  if (!yaml[key]) {
    logger()->error("[YAML] 缺少关键字段: {}", key);
    throw std::runtime_error("Missing key: " + key);
  }
  try {
    return yaml[key].as<T>();
  } catch (const YAML::TypedBadConversion<T> &e) {
    logger()->error("[YAML] 字段 '{}' 类型不匹配: {}", key, e.what());
    throw;
  } catch (const std::exception &e) {
    logger()->error("[YAML] 解析字段 '{}' 失败: {}", key, e.what());
    throw;
  }
}

} // namespace tools
