#include <iostream>
#include <string>
#include <map>
#include <vector>

struct Axis {
    std::string name;
    int extent;
    std::map<std::string, int> strides;
};

int main() {
    Axis a;
    a.name = "a";
    a.extent = 4;
    a.strides["in0"] = 100;
    a.strides["out"] = 200;
    
    int split_factor = 2;
    int original_extent = a.extent;
    a.extent = original_extent / split_factor;
    
    Axis a_in = a; 
    a_in.name = "a_in";
    a_in.extent = split_factor;
    for (auto& [tname, val] : a.strides) {
        val *= split_factor;
    }
    
    std::cout << "a out stride: " << a.strides["out"] << "\n";
    std::cout << "a_in out stride: " << a_in.strides["out"] << "\n";
    for(int i = 0; i < a.extent; ++i) {
        for(int j = 0; j < a_in.extent; ++j) {
            int off = i * a.strides["out"] + j * a_in.strides["out"];
            std::cout << i << "," << j << " -> " << off << "\n";
        }
    }
    return 0;
}
