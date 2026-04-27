//===- tests/parity/load_npy.cpp - Minimal NPY v1.0 reader -----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "load_npy.h"

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ctorch::parity {

namespace {

constexpr std::array<unsigned char, 6> kMagic{0x93, 'N', 'U', 'M', 'P', 'Y'};

dtype descr_to_dtype(const std::string& descr) {
    // The leading '<' / '|' character is the byte-order tag. ctorch only
    // targets little-endian hosts; we accept both orderings as long as the
    // type code matches.
    if (descr == "<f4" || descr == "=f4" || descr == "f4") {
        return dtype::float32;
    }
    if (descr == "<f8" || descr == "=f8" || descr == "f8") {
        return dtype::float64;
    }
    if (descr == "<i4" || descr == "=i4" || descr == "i4") {
        return dtype::int32;
    }
    if (descr == "<i8" || descr == "=i8" || descr == "i8") {
        return dtype::int64;
    }
    if (descr == "|b1" || descr == "b1" || descr == "?") {
        return dtype::bool_;
    }
    throw DTypeError("ctorch::load_npy: dtype '" + descr +
                     "' not yet supported by ctorch::dtype");
}

// Extracts the value associated with a key like `'descr'` from an NPY
// header dict. The parser is intentionally permissive — it just looks for
// `'<key>':` and returns the next token (string, tuple, or bare word).
std::string find_value(const std::string& header, const std::string& key) {
    const std::string needle = "'" + key + "':";
    const std::size_t k = header.find(needle);
    if (k == std::string::npos) {
        throw Error("ctorch::load_npy: NPY header missing '" + key + "'");
    }
    std::size_t i = k + needle.size();
    while (i < header.size() && (header[i] == ' ' || header[i] == '\t')) {
        ++i;
    }
    if (i >= header.size()) {
        throw Error("ctorch::load_npy: NPY header truncated at '" + key + "'");
    }

    // Quoted string.
    if (header[i] == '\'') {
        const std::size_t end = header.find('\'', i + 1);
        if (end == std::string::npos) {
            throw Error("ctorch::load_npy: unterminated string in NPY header");
        }
        return header.substr(i + 1, end - i - 1);
    }
    // Tuple.
    if (header[i] == '(') {
        const std::size_t end = header.find(')', i);
        if (end == std::string::npos) {
            throw Error("ctorch::load_npy: unterminated tuple in NPY header");
        }
        return header.substr(i, end - i + 1);
    }
    // Bare word until a delimiter.
    const std::size_t end = header.find_first_of(",}\n", i);
    return header.substr(i, end - i);
}

std::vector<std::int64_t> parse_shape(const std::string& tuple) {
    // tuple looks like `(3, 4)` or `(5,)` or `()`.
    std::vector<std::int64_t> shape;
    std::size_t i = 0;
    if (i >= tuple.size() || tuple[i] != '(') {
        throw Error("ctorch::load_npy: shape is not a tuple: '" + tuple + "'");
    }
    ++i; // skip '('
    while (i < tuple.size() && tuple[i] != ')') {
        while (i < tuple.size() && (tuple[i] == ' ' || tuple[i] == ',')) {
            ++i;
        }
        if (i >= tuple.size() || tuple[i] == ')') {
            break;
        }
        std::size_t end = i;
        while (end < tuple.size() && tuple[end] != ',' && tuple[end] != ')' &&
               tuple[end] != ' ') {
            ++end;
        }
        const std::string num = tuple.substr(i, end - i);
        try {
            shape.push_back(static_cast<std::int64_t>(std::stoll(num)));
        } catch (...) {
            throw Error("ctorch::load_npy: cannot parse shape integer '" + num + "'");
        }
        i = end;
    }
    return shape;
}

bool parse_bool(const std::string& s) {
    if (s.find("True") != std::string::npos) {
        return true;
    }
    if (s.find("False") != std::string::npos) {
        return false;
    }
    throw Error("ctorch::load_npy: cannot parse bool '" + s + "'");
}

std::int64_t numel_of(const std::vector<std::int64_t>& shape) {
    std::int64_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    return n;
}

} // namespace

Tensor load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error("ctorch::load_npy: cannot open '" + path + "'");
    }

    std::array<unsigned char, 10> head{};
    f.read(reinterpret_cast<char*>(head.data()), 10);
    if (!f) {
        throw Error("ctorch::load_npy: short read on header in '" + path + "'");
    }
    for (std::size_t i = 0; i < kMagic.size(); ++i) {
        if (head[i] != kMagic[i]) {
            throw Error("ctorch::load_npy: bad magic in '" + path + "'");
        }
    }
    const unsigned char major = head[6];
    const unsigned char minor = head[7];
    if (major != 1 || minor != 0) {
        throw Error("ctorch::load_npy: only NPY v1.0 supported (got " +
                    std::to_string(major) + "." + std::to_string(minor) + ")");
    }
    const std::uint16_t header_len =
        static_cast<std::uint16_t>(head[8] | (head[9] << 8));

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (!f) {
        throw Error("ctorch::load_npy: short read on header body in '" + path + "'");
    }

    const std::string descr_s = find_value(header, "descr");
    const std::string fortran_s = find_value(header, "fortran_order");
    const std::string shape_s = find_value(header, "shape");

    const dtype dt = descr_to_dtype(descr_s);
    if (parse_bool(fortran_s)) {
        throw Error("ctorch::load_npy: fortran_order=True is not supported");
    }
    const auto shape = parse_shape(shape_s);

    Tensor out(shape, dt, Device::cpu());
    const std::size_t nbytes =
        static_cast<std::size_t>(numel_of(shape)) * size_of(dt);
    if (nbytes > 0) {
        f.read(static_cast<char*>(out.storage().data()), static_cast<std::streamsize>(nbytes));
        if (!f) {
            throw Error("ctorch::load_npy: short read on data in '" + path + "'");
        }
    }
    return out;
}

} // namespace ctorch::parity
