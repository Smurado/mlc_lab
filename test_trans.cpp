#include <iostream>
#include <vector>
using namespace std;
int main() {
    int extent_a = 96, in_a = 786432, out_a = 192;
    int extent_b = 128, in_b = 6144, out_b = 17664;
    int max1 = (extent_a - 1) * out_a + (extent_b - 1) * out_b; // 95 * 192 + 127 * 17664 = 18240 + 2243328 = 2261568. Plus 31*d = 31*2260992! Wait, OUT sizes are HUGE.
    
    // Original max offset logic using all extents inside get_tensor_size()
    // It doesn't matter what the loops do.
    return 0;
}
