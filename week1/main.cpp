#include <iostream>

// Deklaration der Assembly-Funktion
extern "C" int add(int a, int b);

int main() {
    int result = add(3, 4);
    std::cout << "Result: " << result << std::endl;
    return 0;
}