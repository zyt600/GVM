import re
from pathlib import Path

ENTRY_FILE = "entry_file.h"      # File with CUDA_ENTRY_ENUM(...)
HEADER_FILE = "cuda_api.h"       # File with CUresult CUDAAPI ... declarations

def parse_enum_entries(entry_text):
    pattern = re.compile(r"CUDA_ENTRY_ENUM\((cu\w+)\)")
    return pattern.findall(entry_text)

def extract_function_signature(header_text, func_name):
    # Handles multi-line function declarations
    pattern = re.compile(rf"CUresult\s+CUDAAPI\s+{func_name}\s*\((.*?)\);", re.DOTALL)
    match = pattern.search(header_text)
    if not match:
        return None
    params_block = match.group(1)
    # Remove extra whitespace and split by comma
    params = [p.strip().replace("\n", " ") for p in params_block.split(",") if p.strip()]
    return params

def generate_wrapper(func_name, params):
    # Extract parameter names
    param_names = []
    for i, p in enumerate(params):
        # Try to grab the last word unless it's a pointer or ellipsis
        tokens = p.split()
        name = tokens[-1] if tokens else f"arg{i}"
        name = re.sub(r'[^a-zA-Z0-9_]', '', name) or f'arg{i}'
        param_names.append(name)
    args_call = ", ".join(param_names)
    args_decl = ", ".join(params)
    return (
        f"CUresult {func_name}({args_decl}) {{\n"
        f"  return CUDA_ENTRY_CALL(cuda_library_entry, {func_name}, {args_call});\n"
        f"}}\n"
    )

def main():
    entry_text = Path(ENTRY_FILE).read_text()
    header_text = Path(HEADER_FILE).read_text()

    entries = parse_enum_entries(entry_text)

    for func in entries:
        params = extract_function_signature(header_text, func)
        if params is None:
            print(f"// Signature for {func} not found in header.")
            continue
        wrapper = generate_wrapper(func, params)
        print(wrapper)

if __name__ == "__main__":
    main()
