#include <iostream>

#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;

int main(int argc, char **argv)
{
    std::cout << "hello" << std::endl;
    json j;
    j["hello"] = "world";
    std::string p = j.dump();
    std::cout << p << std::endl;
}