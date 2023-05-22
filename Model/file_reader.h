#pragma once

#include "definitions.h"

#include <fstream>
#include <string>

namespace model {
namespace impl {

class FileReader {
public:
    FileReader(const std::string& filename)
        : fstream_(filename, std::ios_base::in | std::ios_base::out | std::ios::binary) {
    }

    template <typename T>
    void Read(T& result) {
        fstream_.read(reinterpret_cast<char*>(&result), sizeof(result));
    }
    template <typename T>
    void Write(const T& result) {
        fstream_.write(reinterpret_cast<const char*>(&result), sizeof(result));
    }

private:
    std::fstream fstream_;
};

}  // namespace impl
}
