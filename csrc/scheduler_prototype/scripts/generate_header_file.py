import re
import sys

def extract_function_declarations(input_file, output_file):
    # Pattern to match the start of a function definition
    start_pattern = re.compile(r'^(?P<return_type>\w+(?:\s*\*)?)\s+(?P<func_name>\w+)\s*\(', re.MULTILINE)
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("// Auto-generated header file\n")
        outfile.write("// Extracted function declarations\n\n")
        
        buffer = infile.read()
        functions = []
        pos = 0
        
        while True:
            # Find the next function start
            match = start_pattern.search(buffer, pos)
            if not match:
                break
                
            return_type = match.group('return_type')
            func_name = match.group('func_name')
            param_start = match.end()
            
            # Find matching parenthesis
            depth = 1
            param_end = param_start
            while depth > 0 and param_end < len(buffer):
                if buffer[param_end] == '(':
                    depth += 1
                elif buffer[param_end] == ')':
                    depth -= 1
                param_end += 1
            
            if depth != 0:
                pos = param_end
                continue  # Skip unclosed parentheses
                
            # Extract and clean parameters
            params = buffer[param_start:param_end-1]  # -1 to exclude the closing )
            params = re.sub(r'/\*.*?\*/', '', params, flags=re.DOTALL)  # Remove /* */ comments
            params = re.sub(r'//.*$', '', params, flags=re.MULTILINE)   # Remove // comments
            params = ' '.join(params.split())  # Normalize whitespace
            
            # Handle no-parameters case
            if not params.strip():
                params = 'void'
            
            # Save the declaration
            functions.append(f"{return_type} {func_name}({params});")
            pos = param_end
        
        # Write all found functions
        outfile.write("\n".join(functions) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_declarations.py <input_file> <output_header_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    extract_function_declarations(input_file, output_file)
    print(f"Function declarations extracted to {output_file}")