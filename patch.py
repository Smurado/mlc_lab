import re

with open("week8/teir_optimizer.cpp", "r") as f:
    text = f.read()

# Fix einsum
text = re.sub(
r'''            \} else if \(prog\.name == "einsum"\) \{
                set_policy\("b", "parallel"\);
                set_policy\("b", "parallel"\);
                set_policy\("c", "parallel"\);''',
r'''            } else if (prog.name == "einsum") {
                set_policy("a", "parallel");
                set_policy("b", "parallel");
                set_policy("c", "parallel");''', text)

with open("week8/teir_optimizer.cpp", "w") as f:
    f.write(text)
