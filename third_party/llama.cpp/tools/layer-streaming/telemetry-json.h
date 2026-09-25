#pragma once

#include <string>
#include <vector>

namespace ls_json {

std::string value(const std::string & line, const char * key);
std::vector<long long> integer_list(const std::string & line, const char * key, bool * found);
long long integer(const std::string & line, const char * key, bool * found);
std::vector<long long> id_list(const std::string & ids);

} // namespace ls_json
