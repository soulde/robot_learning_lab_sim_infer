#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << "Usage: rll-policy --config <policy.yaml>\n";
    return 0;
  }
  std::cerr << "rll-policy: native runtime is not configured yet; use --help\n";
  return 2;
}
