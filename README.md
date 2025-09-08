
# This project has been discontinued in favor of roto (https://github.com/NLnetLabs/roto)
# fdl
## Fast Deterministic Language

The aim of this language is to provide a deterministic language that can be compiled into bytecode dynamically, and
produces functions that are guaranteed to exit without errors. It does so by limiting the language features, and having
static typing.

# Limitations
 - There are no runtime errors.
   - Division by 0 yields 0 as result
 - All programs need to halt at some point
   - This means no recursive function calls
 - All tables and arrays have to be initialized during construction
 - Arrays have a fixed size, determined during compile time
 - Designed to be used for helper-like functions, and not for big projects
 - Functions can only be called by name, and share a single, global namespace
 - Functions only be defined in the global scope
