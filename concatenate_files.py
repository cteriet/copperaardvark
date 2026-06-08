import argparse
from pathlib import Path


def concatenate_python_files(input_dir: str, output_file: str) -> None:
    """
    Concatenates all Python (.py) files within a given directory and its subdirectories
    into a single output file. A clearly distinguishable header containing the file's
    absolute path is inserted before each file's content.

    Args:
        input_dir (str): The root directory to search for Python files.
        output_file (str): The path to the output file where contents will be written.
        
    Raises:
        NotADirectoryError: If the provided input_dir does not exist or is not a directory.
    """
    root_path = Path(input_dir).resolve()
    out_path = Path(output_file).resolve()

    if not root_path.is_dir():
        raise NotADirectoryError(f"The input directory '{input_dir}' does not exist or is not a directory.")

    # Create the parent directories for the output file if they don't exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as outfile:
        # rglob finds all files matching the pattern recursively
        for py_file in root_path.rglob('*.py'):
            
            # Skip the output file if it happens to be saved in the input directory tree
            if py_file == out_path:
                continue
            
            # Create a highly distinguishable spacer/header
            header = (
                f"\n{'#' * 80}\n"
                f"# FILE: {py_file.absolute()}\n"
                f"{'#' * 80}\n\n"
            )
            
            try:
                with py_file.open('r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(header)
                    outfile.write(content)
                    outfile.write("\n")
            except Exception as e:
                print(f"Warning: Could not read file {py_file}. Error: {e}")
    
    print(f"Successfully concatenated Python files from '{root_path}' into '{out_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate all Python files in a folder (and subfolders) into a single file."
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True, 
        help="The input directory to recursively search for .py files."
    )
    parser.add_argument(
        "-o", "--output-file", type=str, required=True, 
        help="The output file path to save the concatenated content."
    )
    
    args = parser.parse_args()
    concatenate_python_files(args.input_dir, args.output_file)