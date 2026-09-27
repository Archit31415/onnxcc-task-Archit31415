#include "onnxcc/cli/cli.hpp"
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CLIResult {
    int exit_code;
    std::string stdout_str;
    std::string stderr_str;
};

CLIResult run_cli(const std::vector<std::string> &args) {
    std::vector<char *> argv_c;
    std::string prog_name = "onnxcc";
    argv_c.push_back(const_cast<char *>(prog_name.c_str()));
    for (const auto &arg : args) {
        argv_c.push_back(const_cast<char *>(arg.c_str()));
    }

    std::stringstream out_buf;
    std::stringstream err_buf;

    std::streambuf *old_out = std::cout.rdbuf(out_buf.rdbuf());
    std::streambuf *old_err = std::cerr.rdbuf(err_buf.rdbuf());

    int code = onnxcc::cli::run(static_cast<int>(argv_c.size()), argv_c.data());

    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);

    return {code, out_buf.str(), err_buf.str()};
}

} // namespace

TEST(CLITest, DumpWithModelPath) {
    auto res = run_cli({"dump", "--model", "path/to/file.onnx"});
    EXPECT_EQ(res.exit_code, 0);
    EXPECT_TRUE(res.stderr_str.empty());
}

TEST(CLITest, DumpWithShowGraph) {
    auto res = run_cli({"dump", "--model", "f.onnx", "--show-graph"});
    EXPECT_EQ(res.exit_code, 0);
    EXPECT_TRUE(res.stderr_str.empty());
}

TEST(CLITest, DumpWithVerbose) {
    auto res = run_cli({"dump", "--model", "f.onnx", "--verbose"});
    EXPECT_EQ(res.exit_code, 0);
    EXPECT_TRUE(res.stderr_str.empty());
}

TEST(CLITest, DumpHelp) {
    auto res = run_cli({"dump", "--help"});
    EXPECT_EQ(res.exit_code, 0);
    EXPECT_NE(res.stdout_str.find("Usage"), std::string::npos);
    EXPECT_NE(res.stdout_str.find("model"), std::string::npos);
    EXPECT_NE(res.stdout_str.find("show-graph"), std::string::npos);
    EXPECT_NE(res.stdout_str.find("verbose"), std::string::npos);
}

TEST(CLITest, TopLevelHelp) {
    auto res = run_cli({"--help"});
    EXPECT_EQ(res.exit_code, 0);
    EXPECT_NE(res.stdout_str.find("Usage"), std::string::npos);
    EXPECT_NE(res.stdout_str.find("dump"), std::string::npos);
}

TEST(CLITest, DumpWithoutModelMissingArg) {
    auto res = run_cli({"dump"});
    EXPECT_NE(res.exit_code, 0);
    EXPECT_FALSE(res.stderr_str.empty());
    EXPECT_NE(res.stderr_str.find("--model"), std::string::npos);
}

TEST(CLITest, UnknownSubcommand) {
    auto res = run_cli({"bogus"});
    EXPECT_NE(res.exit_code, 0);
    EXPECT_FALSE(res.stderr_str.empty());
    EXPECT_NE(res.stderr_str.find("bogus"), std::string::npos);
}

TEST(CLITest, NoArgumentsPassed) {
    auto res = run_cli({});
    EXPECT_NE(res.exit_code, 0);
    EXPECT_FALSE(res.stderr_str.empty());
    EXPECT_NE(res.stderr_str.find("Usage"), std::string::npos);
}
