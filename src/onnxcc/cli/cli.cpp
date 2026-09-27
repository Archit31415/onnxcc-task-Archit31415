#include "onnxcc/cli/cli.hpp"
#include "onnxcc/third_party/cxxopts.hpp"

#include <iostream>
#include <string>

namespace onnxcc::cli {

namespace {

void print_top_level_help(std::ostream &os) {
    os << "Usage:\n"
       << "  onnxcc <subcommand> [options]\n\n"
       << "Available subcommands:\n"
       << "  dump       Dump information about an ONNX model\n"
       << "  run        Run inference on an ONNX model\n"
       << "  compile    Compile an ONNX model\n"
       << "  benchmark  Benchmark an ONNX model\n\n"
       << "Use \"onnxcc <subcommand> --help\" for more information on a subcommand.\n";
}

int handle_dump_subcommand(int argc, char **argv) {
    try {
        cxxopts::Options options("onnxcc dump", "Dump information about an ONNX model");
        options.add_options()
            ("m,model", "Path to input ONNX model file", cxxopts::value<std::string>())
            ("show-graph", "Display model computational graph", cxxopts::value<bool>()->default_value("false"))
            ("verbose", "Enable verbose output", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print help for dump subcommand");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        if (!result.count("model") || result["model"].as<std::string>().empty()) {
            std::cerr << "Error: --model option is required for dump subcommand.\n";
            return 1;
        }

        return 0;
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << "Error parsing dump options: " << e.what() << "\n";
        return 1;
    }
}

} // namespace

int run(int argc, char **argv) {
    if (argc <= 1) {
        print_top_level_help(std::cerr);
        return 1;
    }

    std::string subcommand = argv[1];

    if (subcommand == "--help" || subcommand == "-h") {
        print_top_level_help(std::cout);
        return 0;
    }

    if (subcommand == "dump") {
        return handle_dump_subcommand(argc - 1, argv + 1);
    }

    std::cerr << "Error: Unknown subcommand '" << subcommand << "'.\n";
    print_top_level_help(std::cerr);
    return 1;
}

} // namespace onnxcc::cli
