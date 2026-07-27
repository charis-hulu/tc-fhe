#pragma once

#ifdef TC_WITH_TFHE

#include <tfhe.h>

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace btc {

// Loads a BooleanParameters struct from a flat JSON file, e.g.:
//
//   {
//     "lwe_dimension": 805,
//     "glwe_dimension": 3,
//     "polynomial_size": 512,
//     "lwe_noise_distribution": { "gaussian_std_dev": 5.8615896642671336e-06 },
//     "glwe_noise_distribution": { "gaussian_std_dev": 9.315272083503367e-10 },
//     "pbs_base_log": 10,
//     "pbs_level": 2,
//     "ks_base_log": 3,
//     "ks_level": 5,
//     "encryption_key_choice": "small"
//   }
//
// Noise distributions may instead use "t_uniform_bound_log2" for TUniform noise.
// This is a purpose-built reader for this one flat schema, not a general JSON
// parser: no nested arrays, no escaping beyond what's needed for the fields above.
inline BooleanParameters load_boolean_parameters(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_boolean_parameters: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();

    std::size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    };
    auto expect = [&](char c) {
        skip_ws();
        if (pos >= text.size() || text[pos] != c)
            throw std::runtime_error("load_boolean_parameters: expected '" + std::string(1, c) +
                                      "' at offset " + std::to_string(pos));
        ++pos;
    };
    auto parse_string = [&]() -> std::string {
        skip_ws();
        expect('"');
        std::string out;
        while (pos < text.size() && text[pos] != '"') {
            if (text[pos] == '\\' && pos + 1 < text.size()) ++pos;
            out += text[pos];
            ++pos;
        }
        expect('"');
        return out;
    };
    auto parse_number = [&]() -> double {
        skip_ws();
        std::size_t start = pos;
        while (pos < text.size() &&
               (std::isdigit(static_cast<unsigned char>(text[pos])) || text[pos] == '-' ||
                text[pos] == '+' || text[pos] == '.' || text[pos] == 'e' || text[pos] == 'E'))
            ++pos;
        if (pos == start)
            throw std::runtime_error("load_boolean_parameters: expected number at offset " +
                                      std::to_string(start));
        return std::stod(text.substr(start, pos - start));
    };
    // Parses either a number or a single-key noise-distribution object;
    // returns the object's single key/value as a synthetic "key:value" pair,
    // or the number itself with key "".
    struct NoiseSpec { bool is_t_uniform = false; double value = 0.0; };
    auto parse_noise_distribution = [&]() -> NoiseSpec {
        skip_ws();
        expect('{');
        skip_ws();
        std::string key = parse_string();
        expect(':');
        double v = parse_number();
        skip_ws();
        expect('}');
        NoiseSpec spec;
        if (key == "gaussian_std_dev") {
            spec.is_t_uniform = false;
            spec.value = v;
        } else if (key == "t_uniform_bound_log2") {
            spec.is_t_uniform = true;
            spec.value = v;
        } else {
            throw std::runtime_error("load_boolean_parameters: unknown noise distribution key '" +
                                      key + "'");
        }
        return spec;
    };

    std::unordered_map<std::string, double> numbers;
    std::unordered_map<std::string, NoiseSpec> noise;
    std::unordered_map<std::string, std::string> strings;

    expect('{');
    skip_ws();
    if (pos < text.size() && text[pos] != '}') {
        while (true) {
            skip_ws();
            std::string key = parse_string();
            expect(':');
            skip_ws();
            if (pos < text.size() && text[pos] == '{') {
                noise[key] = parse_noise_distribution();
            } else if (pos < text.size() && text[pos] == '"') {
                strings[key] = parse_string();
            } else {
                numbers[key] = parse_number();
            }
            skip_ws();
            if (pos < text.size() && text[pos] == ',') { ++pos; continue; }
            break;
        }
    }
    expect('}');

    auto require_number = [&](const std::string& key) -> std::size_t {
        auto it = numbers.find(key);
        if (it == numbers.end())
            throw std::runtime_error("load_boolean_parameters: missing field '" + key + "'");
        return static_cast<std::size_t>(it->second);
    };
    auto require_noise = [&](const std::string& key) -> DynamicDistribution {
        auto it = noise.find(key);
        if (it == noise.end())
            throw std::runtime_error("load_boolean_parameters: missing field '" + key + "'");
        return it->second.is_t_uniform
                   ? new_t_uniform(static_cast<uint32_t>(it->second.value))
                   : new_gaussian_from_std_dev(it->second.value);
    };

    BooleanParameters params{};
    params.lwe_dimension = require_number("lwe_dimension");
    params.glwe_dimension = require_number("glwe_dimension");
    params.polynomial_size = require_number("polynomial_size");
    params.lwe_noise_distribution = require_noise("lwe_noise_distribution");
    params.glwe_noise_distribution = require_noise("glwe_noise_distribution");
    params.pbs_base_log = require_number("pbs_base_log");
    params.pbs_level = require_number("pbs_level");
    params.ks_base_log = require_number("ks_base_log");
    params.ks_level = require_number("ks_level");

    auto key_choice_it = strings.find("encryption_key_choice");
    if (key_choice_it == strings.end())
        throw std::runtime_error("load_boolean_parameters: missing field 'encryption_key_choice'");
    if (key_choice_it->second == "big")
        params.encryption_key_choice = BooleanEncryptionKeyChoiceBig;
    else if (key_choice_it->second == "small")
        params.encryption_key_choice = BooleanEncryptionKeyChoiceSmall;
    else
        throw std::runtime_error(
            "load_boolean_parameters: encryption_key_choice must be \"big\" or \"small\"");

    return params;
}

} // namespace btc

#endif // TC_WITH_TFHE
