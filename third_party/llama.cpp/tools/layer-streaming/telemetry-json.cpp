#include "telemetry-json.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>

namespace ls_json {

std::string value(const std::string & line, const char * key) {
    const std::string pattern = std::string("\"") + key + "\":";
    const size_t pos = line.find(pattern);
    if (pos == std::string::npos) {
        return "";
    }
    size_t start = pos + pattern.size();
    while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
        start++;
    }
    if (start < line.size() && line[start] == '"') {
        start++;
        size_t end = start;
        while (end < line.size() && line[end] != '"') {
            end++;
        }
        return line.substr(start, end - start);
    }
    size_t end = start;
    while (end < line.size() && line[end] != ',' && line[end] != '}') {
        end++;
    }
    return line.substr(start, end - start);
}

std::vector<long long> integer_list(const std::string & line, const char * key, bool * found) {
    std::vector<long long> out;
    const std::string pattern = std::string("\"") + key + "\"";
    const size_t key_pos = line.find(pattern);
    if (key_pos == std::string::npos) {
        if (found) {
            *found = false;
        }
        return out;
    }
    size_t index = key_pos + pattern.size();
    while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
        index++;
    }
    if (index >= line.size() || line[index] != ':') {
        if (found) {
            *found = false;
        }
        return out;
    }
    index++;
    while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
        index++;
    }
    if (index >= line.size() || line[index] != '[') {
        if (found) {
            *found = false;
        }
        return out;
    }
    index++;
    while (index < line.size()) {
        while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
            index++;
        }
        if (index < line.size() && line[index] == ']') {
            index++;
            while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
                index++;
            }
            if (index < line.size() && line[index] != ',' && line[index] != '}') {
                if (found) {
                    *found = false;
                }
                return {};
            }
            if (found) {
                *found = true;
            }
            return out;
        }
        const size_t token_start = index;
        if (line[index] == '-') {
            index++;
        }
        const size_t digits = index;
        while (index < line.size() && line[index] >= '0' && line[index] <= '9') {
            index++;
        }
        if (index == digits) {
            if (found) {
                *found = false;
            }
            return {};
        }
        const std::string token = line.substr(token_start, index - token_start);
        char * end = nullptr;
        errno = 0;
        const long long parsed = strtoll(token.c_str(), &end, 10);
        if (errno == ERANGE || end == token.c_str() || *end != '\0') {
            if (found) {
                *found = false;
            }
            return {};
        }
        out.push_back(parsed);
        while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
            index++;
        }
        if (index >= line.size()) {
            if (found) {
                *found = false;
            }
            return {};
        }
        if (line[index] == ',') {
            index++;
            continue;
        }
        if (line[index] != ']') {
            if (found) {
                *found = false;
            }
            return {};
        }
    }
    if (found) {
        *found = false;
    }
    return {};
}

long long integer(const std::string & line, const char * key, bool * found) {
    const std::string pattern = std::string("\"") + key + "\"";
    const size_t key_pos = line.find(pattern);
    if (key_pos == std::string::npos) {
        if (found) {
            *found = false;
        }
        return 0;
    }
    size_t index = key_pos + pattern.size();
    while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
        index++;
    }
    if (index >= line.size() || line[index] != ':') {
        if (found) {
            *found = false;
        }
        return 0;
    }
    index++;
    while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
        index++;
    }
    const size_t start = index;
    while (index < line.size() && line[index] != ',' && line[index] != '}') {
        index++;
    }
    std::string text = line.substr(start, index - start);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) {
        text.pop_back();
    }
    char * end = nullptr;
    errno = 0;
    // Telemetry phase timings and byte counts are emitted as JSON numbers and
    // some producer paths serialize integral-valued fields with a decimal
    // suffix (for example 610797.0). Keep the old truncation behavior for
    // those fields, but reject quotes, garbage, and partially parsed tokens.
    const long double parsed = strtold(text.c_str(), &end);
    if (text.empty() || errno == ERANGE || end == text.c_str() || *end != '\0' ||
        !std::isfinite(parsed) ||
        parsed < static_cast<long double>(LLONG_MIN) ||
        parsed > static_cast<long double>(LLONG_MAX)) {
        if (found) {
            *found = false;
        }
        return 0;
    }
    if (found) {
        *found = true;
    }
    return static_cast<long long>(parsed);
}

std::vector<long long> id_list(const std::string & ids) {
    std::vector<long long> out;
    size_t pos = 0;
    while (pos <= ids.size()) {
        const size_t comma = ids.find(',', pos);
        const std::string token = ids.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (token.empty()) {
            break;
        }
        char * end = nullptr;
        const long long parsed = strtoll(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0') {
            out.clear();
            return out;
        }
        out.push_back(parsed);
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

} // namespace ls_json
